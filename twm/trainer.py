import multiprocessing
import os
import time
from tqdm import tqdm

import gym
import numpy as np
import torch
import torchvision
from PIL import ImageDraw

import wandb
from agent import Agent, Dreamer
from replay_buffer import ReplayBuffer

import utils

import matplotlib.pyplot as plt


class Trainer:

    def __init__(self, config):
        if config['buffer_prefill'] <= 0:
            raise ValueError()
        if config['pretrain_obs_p'] + config['pretrain_dyn_p'] > 1:
            raise ValueError()

        self.config = config
        self.env = self._create_env_from_config(config)
        self.replay_buffer = ReplayBuffer(config, self.env)

        num_actions = self.env.action_space.n if isinstance(self.env.action_space,gym.spaces.Discrete) \
                        else np.prod(self.env.action_space.shape)
        self.action_meanings = self.env.get_action_meanings()
        self.agent = Agent(config, num_actions).to(config['model_device'])

        # metrics that won't be summarized, the last value will be used instead
        except_keys = ['buffer/size', 'buffer/total_reward', 'buffer/num_episodes']
        self.summarizer = utils.MetricsSummarizer(except_keys=except_keys)
        self.last_eval = 0
        self.total_eval_time = 0

    def print_stats(self):
        count_params = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
        agent = self.agent
        wm = agent.wm
        ac = agent.ac
        print('# Parameters')
        print('Observation model:', count_params(wm.obs_model))
        print('Dynamics model:', count_params(wm.dyn_model))
        print('Actor:', count_params(ac.actor_model))
        print('Critic:', count_params(ac.critic_model))
        print('World model:', count_params(wm))
        print('Actor-critic:', count_params(ac))
        print('Observation encoder + actor:', count_params(wm.obs_model.encoder) + count_params(ac.actor_model))
        print('Total:', count_params(agent))

    def close(self):
        self.env.close()

    @staticmethod
    def _create_env_from_config(config, eval=False):
        if config['env_suite'] == 'atari':
            noop_max = 0 if eval else config['env_noop_max']
            env = utils.create_atari_env(
                config['game'], noop_max, config['env_frame_skip'], config['env_frame_stack'],
                config['env_frame_size'], config['env_episodic_lives'], config['env_grayscale'], config['env_time_limit'])
        elif config['env_suite'] == 'd4rl':
            env = utils.create_d4rl_env(config['game'], config['env_render_mode'])
        elif config['env_suite'] == 'd3rl':
            env = utils.create_d3rl_env(config['game'], 
                                        render_mode=config['env_render_mode'], 
                                        frame_stack=config['env_frame_stack'],
                                        max_episode_steps=config['env_time_limit'],
                                        )
        else:
            raise NotImplementedError(f'Invalid Environment Suite: {config["env_suite"]}')
        if eval:
            env = utils.NoAutoReset(env)  # must use options={'force': True} to really reset
        return env

    def _create_buffer_obs_policy(self):
        # actor-critic policy acting on buffer data at index
        config = self.config
        agent = self.agent
        device = next(agent.parameters()).device
        wm = agent.wm
        ac = agent.ac
        replay_buffer = self.replay_buffer
        dreamer = None

        @torch.no_grad()
        def policy(index):
            nonlocal dreamer
            if dreamer is None:
                prefix = config['wm_memory_length'] - 1
                start_o = replay_buffer.get_obs([[index]], device=device, prefix=prefix + 1)
                start_a = replay_buffer.get_actions([[index - 1]], device=device, prefix=prefix)
                start_r = replay_buffer.get_rewards([[index - 1]], device=device, prefix=prefix)
                start_terminated = replay_buffer.get_terminated([[index - 1]], device=device, prefix=prefix)
                start_truncated = replay_buffer.get_truncated([[index - 1]], device=device, prefix=prefix)

                dreamer = Dreamer(config, wm, mode='observe', ac=ac, store_data=False)
                dreamer.observe_reset(start_o, start_a, start_r, start_terminated, start_truncated)
                a = dreamer.act()
            else:
                o = replay_buffer.get_obs([[index]], device=device)
                a = replay_buffer.get_actions([[index - 1]], device=device)
                r = replay_buffer.get_rewards([[index - 1]], device=device)
                terminated = replay_buffer.get_terminated([[index - 1]], device=device)
                truncated = replay_buffer.get_truncated([[index - 1]], device=device)
                dreamer.observe_step(a, o, r, terminated, truncated)
                a = dreamer.act()
            a = a.squeeze()
            if not a.ndim:
                a = a.item()
            else:
                a = a.cpu().numpy()
            return a
        return policy

    def _create_start_z_sampler(self, temperature):
        obs_model = self.agent.wm.obs_model
        replay_buffer = self.replay_buffer

        @torch.no_grad()
        def sampler(n):
            idx = utils.random_choice(replay_buffer.size, n, device=replay_buffer.device)
            o = replay_buffer.get_obs(idx).unsqueeze(1)
            z = obs_model.eval().encode_sample(o, temperature=temperature).squeeze(1)
            return z

        return sampler

    def run(self):
        config = self.config
        replay_buffer = self.replay_buffer

        log_every = 20
        self.last_eval = 0
        self.total_eval_time = 0

        # prefill the buffer with randomly collected data
        random_policy = lambda index: replay_buffer.sample_random_action()
        for _ in tqdm(range(config['buffer_prefill'] - 1), desc=f'Prefilling buffer (random policy)'):
            replay_buffer.step(random_policy)
            metrics = {}
            utils.update_metrics(metrics, replay_buffer.metrics(), prefix='buffer/')
            self.summarizer.append(metrics)
            if replay_buffer.size % log_every == 0:
                wandb.log(self.summarizer.summarize())

        # final prefill step
        replay_buffer.step(random_policy)
        metrics = {}
        utils.update_metrics(metrics, replay_buffer.metrics(), prefix='buffer/')

        # pretrain on the prefilled data
        self._pretrain()

        eval_metrics = self._evaluate(is_final=False)
        metrics.update(eval_metrics)
        self.summarizer.append(metrics)
        wandb.log(self.summarizer.summarize())

        budget = config['budget'] - config['pretrain_budget']
        budget_per_step = 0
        budget_per_step += config['wm_train_steps'] * config['wm_batch_size'] * config['wm_sequence_length']
        budget_per_step += config['ac_batch_size'] * config['ac_horizon']
        num_batches = budget / budget_per_step
        train_every = (replay_buffer.capacity - config['buffer_prefill']) / num_batches

        step_counter = 0
        with tqdm(total=replay_buffer.capacity,desc='Training TWM') as pbar:
            while replay_buffer.size < replay_buffer.capacity:
                init_size = replay_buffer.size
                # collect data in real environment
                collect_policy = self._create_buffer_obs_policy()
                should_log = False
                with tqdm(total=train_every,desc='Collect data',initial=step_counter,leave=False) as _pbar:
                    while step_counter <= train_every and replay_buffer.size < replay_buffer.capacity:
                        if replay_buffer.size - self.last_eval >= config['eval_every']:
                            metrics = self._evaluate(is_final=False)
                            utils.update_metrics(metrics, replay_buffer.metrics(), prefix='buffer/')
                            self.summarizer.append(metrics)
                            wandb.log(self.summarizer.summarize())

                        replay_buffer.step(collect_policy)
                        step_counter += 1

                        if replay_buffer.size % log_every == 0:
                            should_log = True
                        _pbar.update(1)

                # train world model and actor-critic
                metrics_hist = []
                with tqdm(total=step_counter,desc='Train wm/ac',initial=step_counter,leave=False) as _pbar:
                    while step_counter >= train_every:
                        step_counter -= train_every
                        metrics = self._train_step()
                        metrics_hist.append(metrics)
                        _pbar.update(train_every)

                metrics = utils.mean_metrics(metrics_hist)
                utils.update_metrics(metrics, replay_buffer.metrics(), prefix='buffer/')

                # evaluate
                if replay_buffer.size - self.last_eval >= config['eval_every'] and \
                        replay_buffer.size < replay_buffer.capacity:
                    eval_metrics = self._evaluate(is_final=False)
                    metrics.update(eval_metrics)
                    should_log = True

                self.summarizer.append(metrics)
                if should_log:
                    wandb.log(self.summarizer.summarize())
                pbar.update(replay_buffer.size-init_size)

                if config['save'] and replay_buffer.size - self.last_eval >= config['save_every'] and \
                        replay_buffer.size < replay_buffer.capacity:
                    filename = 'agent.pt'
                    checkpoint = {'config': dict(config), 'state_dict': self.agent.state_dict(), 'size': replay_buffer.size}
                    torch.save(checkpoint, os.path.join(wandb.run.dir, filename))
                    wandb.save(filename)

        # final evaluation
        metrics = self._evaluate(is_final=True)
        utils.update_metrics(metrics, replay_buffer.metrics(), prefix='buffer/')
        self.summarizer.append(metrics)
        wandb.log(self.summarizer.summarize())

        # save final model
        if config['save']:
            filename = 'agent_final.pt'
            checkpoint = {'config': dict(config), 'state_dict': self.agent.state_dict(), 'size': replay_buffer.size}
            torch.save(checkpoint, os.path.join(wandb.run.dir, filename))
            wandb.save(filename)

    def _pretrain(self):
        config = self.config
        agent = self.agent
        device = next(agent.parameters()).device
        wm = agent.wm
        obs_model = wm.obs_model
        ac = agent.ac
        replay_buffer = self.replay_buffer

        # pretrain observation model
        wm_total_batch_size = config['wm_batch_size'] * config['wm_sequence_length']
        budget = config['pretrain_budget'] * config['pretrain_obs_p']
        with tqdm(total=budget,desc='Pretraining observation model') as pbar:
            while budget > 0:
                indices = torch.randperm(replay_buffer.size, device=replay_buffer.device)
                while len(indices) > 0 and budget > 0:
                    idx = indices[:wm_total_batch_size]
                    indices = indices[wm_total_batch_size:]
                    o = replay_buffer.get_obs(idx, device=device)
                    _ = wm.optimize_pretrain_obs(o.unsqueeze(1))
                    budget -= idx.numel()
                    pbar.update(idx.numel())

        # encode all observations once, since the encoder does not change anymore
        indices = torch.arange(replay_buffer.size, dtype=torch.long, device=replay_buffer.device)
        o = replay_buffer.get_obs(indices.unsqueeze(0), prefix=1, device=device, return_next=True)  # 1 for context
        o = o.squeeze(0).unsqueeze(1)
        with torch.no_grad():
            z_dist = obs_model.eval().encode(o)

        # pretrain dynamics model
        budget = config['pretrain_budget'] * config['pretrain_dyn_p']
        with tqdm(total=budget,desc='Pretraining dynamics model') as pbar:
            while budget > 0:
                for idx in replay_buffer.generate_uniform_indices(
                        config['wm_batch_size'], config['wm_sequence_length'], extra=2):  # 2 for context + next
                    z, logits = obs_model.sample_z(z_dist, idx=idx.flatten(), return_logits=True)
                    z, logits = [x.squeeze(1).unflatten(0, idx.shape) for x in (z, logits)]
                    z = z[:, :-1]
                    target_logits = logits[:, 2:]
                    idx = idx[:, :-2]
                    _, a, r, terminated, truncated, _ = replay_buffer.get_data(idx, device=device, prefix=1)
                    _ = wm.optimize_pretrain_dyn(z, a, r, terminated, truncated, target_logits)
                    budget -= idx.numel()
                    if budget <= 0:
                        break
                    pbar.update(idx.numel())

        # pretrain ac
        budget = config['pretrain_budget'] * (1 - config['pretrain_obs_p'] + config['pretrain_dyn_p'])
        with tqdm(total=budget,desc='Pretraining actor-critic') as pbar:
            while budget > 0:
                for idx in replay_buffer.generate_uniform_indices(
                        config['ac_batch_size'], config['ac_horizon'], extra=2):  # 2 for context + next
                    z = obs_model.sample_z(z_dist, idx=idx.flatten())
                    z = z.squeeze(1).unflatten(0, idx.shape)
                    idx = idx[:, :-2]
                    _, a, r, terminated, truncated, _ = replay_buffer.get_data(idx, device=device, prefix=1)
                    d = torch.logical_or(terminated, truncated)
                    if config['ac_input_h']:
                        g = wm.to_discounts(terminated)
                        tgt_length = config['ac_horizon'] + 1
                        with torch.no_grad():
                            _, h, _ = wm.dyn_model.eval().predict(z[:, :-1], a, r, g, d[:, :-1], tgt_length)
                    else:
                        h = None
                    g = wm.to_discounts(d)
                    z, r, g, d = [x[:, 1:] for x in (z, r, g, d)]
                    _ = ac.optimize_pretrain(z, h, r, g, d)
                    budget -= idx.numel()
                    if budget <= 0:
                        break
                    pbar.update(idx.numel())
            ac.sync_target()

    def _train_step(self):
        config = self.config
        agent = self.agent
        device = next(agent.parameters()).device
        wm = agent.wm
        ac = agent.ac
        replay_buffer = self.replay_buffer

        # train wm
        for _ in tqdm(range(config['wm_train_steps']),desc='Train wm',leave=False):
            metrics_i = {}
            idx = replay_buffer.sample_indices(config['wm_batch_size'], config['wm_sequence_length'])
            o, a, r, terminated, truncated, _ = \
                replay_buffer.get_data(idx, device=device, prefix=1, return_next_obs=True)  # 1 for context

            z, h, met = wm.optimize(o, a, r, terminated, truncated)
            utils.update_metrics(metrics_i, met, prefix='wm/')

            o, a, r, terminated, truncated = [x[:, :-1] for x in (o, a, r, terminated, truncated)]

        metrics = metrics_i  # only use last metrics

        # train actor-critic
        create_start = lambda x, size: utils.windows(x, size).flatten(0, 1)
        start_z = create_start(z, 2)
        start_a = create_start(a, 1)
        start_r = create_start(r, 1)
        start_terminated = create_start(terminated, 1)
        start_truncated = create_start(truncated, 1)

        idx = utils.random_choice(start_z.shape[0], config['ac_batch_size'], device=start_z.device)
        start_z, start_a, start_r, start_terminated, start_truncated = \
            [x[idx] for x in (start_z, start_a, start_r, start_terminated, start_truncated)]

        dreamer = Dreamer(config, wm, mode='imagine', ac=ac, store_data=True,
                          start_z_sampler=self._create_start_z_sampler(temperature=1))
        dreamer.imagine_reset(start_z, start_a, start_r, start_terminated, start_truncated)
        for _ in tqdm(range(config['ac_horizon']),desc='Train ac',leave=False):
            a = dreamer.act()
            dreamer.imagine_step(a)
        z, o, h, a, r, g, d, weights = dreamer.get_data()
        if config['wm_discount_threshold'] == 0:
            d = None  # save some computation, since all dones are False in this case
        ac_metrics = ac.optimize(z, h, a, r, g, d, weights)
        utils.update_metrics(metrics, ac_metrics, prefix='ac/')

        return metrics

    @torch.no_grad()
    def _evaluate(self, is_final):
        print(f'Running{" FINAL " if is_final else " "}evaluation...')
        start_time = time.time()
        config = self.config
        agent = self.agent
        device = next(agent.parameters()).device
        wm = agent.wm
        ac = agent.ac
        replay_buffer = self.replay_buffer

        metrics = {}
        metrics['buffer/visits'] = replay_buffer.visit_histogram()
        metrics['buffer/sample_probs'] = replay_buffer.sample_probs_histogram()
        recon_img, imagine_img = self._create_eval_images(is_final, config['env_suite'])
        if recon_img is not None:
            metrics['eval/recons'] = wandb.Image(recon_img)
        if imagine_img is not None:
            metrics['eval/imagine'] = wandb.Image(imagine_img)

        # similar to evaluation proposed in https://arxiv.org/pdf/2007.05929.pdf (SPR) section 4.1
        num_episodes = config['final_eval_episodes'] if is_final else config['eval_episodes']
        num_envs = max(min(num_episodes, int(multiprocessing.cpu_count() * config['cpu_p'])), 1)
        env_fn = lambda: Trainer._create_env_from_config(config, eval=True)
        eval_env = utils.create_vector_env(num_envs, env_fn)

        seed = ((config['seed'] + 13) * 7919 + 13) if config['seed'] is not None else None
        start_obs, _ = eval_env.reset(seed=seed)
        start_obs = utils.preprocess_obs(start_obs, device, config['env_suite']).unsqueeze(1)

        dreamer = Dreamer(config, wm, mode='observe', ac=ac, store_data=False)
        dreamer.observe_reset_single(start_obs)

        scores = []
        current_scores = np.zeros(num_envs)
        finished = np.zeros(num_envs, dtype=bool)
        num_truncated = 0
        with tqdm(total=num_episodes, desc='Evaluating', leave=False) as pbar:
            while len(scores) < num_episodes:
                a = dreamer.act()
                o, r, terminated, truncated, infos = eval_env.step(a.squeeze(1).cpu().numpy())

                not_finished = ~finished
                current_scores[not_finished] += r[not_finished]
                lives = infos.get('lives',np.zeros_like(r))
                for i in range(num_envs):
                    if not finished[i]:
                        if truncated[i]:
                            num_truncated += 1
                            finished[i] = True
                        elif terminated[i] and lives[i] == 0:
                            finished[i] = True

                o = utils.preprocess_obs(o, device, config['env_suite']).unsqueeze(1)
                r = torch.as_tensor(r, dtype=torch.float, device=device).unsqueeze(1)
                terminated = torch.as_tensor(terminated, device=device).unsqueeze(1)
                truncated = torch.as_tensor(truncated, device=device).unsqueeze(1)
                z, h, _, d, _ = dreamer.observe_step(a, o, r, terminated, truncated)

                if np.all(finished):
                    # only reset if all environments are finished to remove bias for shorter episodes
                    scores.extend(current_scores.tolist())
                    num_scores = len(scores)
                    if num_scores >= num_episodes:
                        if num_scores > num_episodes:
                            scores = scores[:num_episodes]  # unbiased, just pick first
                        break
                    pbar.update(len(current_scores))
                    current_scores[:] = 0
                    finished[:] = False
                    if seed is not None:
                        seed = seed * 3 + 13 + num_envs
                    start_o, _ = eval_env.reset(seed=seed, options={'force': True})
                    start_o = utils.preprocess_obs(start_o, device, config['env_suite']).unsqueeze(1)
                    dreamer = Dreamer(config, wm, mode='observe', ac=ac, store_data=False)
                    dreamer.observe_reset_single(start_o)
        eval_env.close(terminate=True)
        if num_truncated > 0:
            print(f'{num_truncated} episode(s) truncated')

        score_mean = np.mean(scores)
        score_metrics = {
            'score_mean': score_mean,
            'score_std': np.std(scores),
            'score_median': np.median(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'hns': utils.compute_atari_hns(config['game'], score_mean) if config['env_suite'] == 'atari' else score_mean,
        }
        metrics.update({f'eval/{key}': value for key, value in score_metrics.items()})
        if is_final:
            metrics.update({f'eval/final_{key}': value for key, value in score_metrics.items()})

        end_time = time.time()
        eval_time = end_time - start_time

        self.total_eval_time += eval_time
        metrics['eval/total_time'] = self.total_eval_time

        self.last_eval = replay_buffer.size
        return metrics

    def _create_eval_images(self, is_final=False, suite='atari'):
        if suite == 'atari':
            return self._create_eval_images_atari(is_final)
        elif suite == 'd4rl':
            return self._create_eval_images_d4rl(is_final)
        elif suite == 'd3rl':
            return self._create_eval_images_d3rl(is_final)
        else:
            raise NotImplementedError(f'Unrecognized Environment Suite: {suite}!')

    @torch.no_grad()
    def _create_eval_images_d3rl(self, is_final=False):
        return None, None

    @torch.no_grad()
    def _create_eval_images_d4rl(self, is_final=False):
        return None, None
        config = self.config
        agent = self.agent
        replay_buffer = self.replay_buffer
        obs_model = agent.wm.obs_model.eval()

        # recon_img
        idx = utils.random_choice(replay_buffer.size, 10, device=replay_buffer.device).unsqueeze(1)
        o = replay_buffer.get_obs(idx)
        z = obs_model.encode_sample(o, temperature=0)
        recons = obs_model.decode(z)
        # use last frame of frame stack
        o = o.squeeze(1)
        recons = recons.squeeze(1)
        # recon_img = torch.cat(recon_img, dim=0).squeeze(1).transpose(0, 1).flatten(0, 1)
        # recon_img = torchvision.utils.make_grid(recon_img, nrow=o.shape[0], padding=2)
        # recon_img = utils.to_image(recon_img)

        fig,axes = plt.subplots(10,1,figsize=(16,12), sharex=True)
        fig.suptitle('Reconstruction Comparison')
        x,width = np.arange(len(obs)), 0.25
        for ax,obs,recon in zip(axes,o,recons):
            obs_rects = ax.bar(x,obs,width,label='Observation')
            ax.bar_label(obs_rects, padding=3)
            recon_rects = ax.bar(x + width,recon,width,label='Reconstruction')
            ax.bar_label(recon_rects, padding=3)
        axes[0].legend(loc='upper left', ncols=2)
        axes[-1].set_xticks(x + width, self.env.get_action_meanings())


        # # imagine_img
        # idx = idx[:5]
        # start_o = replay_buffer.get_obs(idx, prefix=1)  # 1 for context
        # start_a = replay_buffer.get_actions(idx, prefix=1)[:, :-1]
        # start_r = replay_buffer.get_rewards(idx, prefix=1)[:, :-1]
        # start_terminated = replay_buffer.get_terminated(idx, prefix=1)[:, :-1]
        # start_truncated = replay_buffer.get_truncated(idx, prefix=1)[:, :-1]
        # start_z = obs_model.encode_sample(start_o, temperature=0)

        # horizon = 100 if is_final else config['wm_sequence_length']
        # dreamer = Dreamer(config, agent.wm, mode='imagine', ac=agent.ac, store_data=True,
        #                   start_z_sampler=self._create_start_z_sampler(temperature=0), always_compute_obs=True)
        # dreamer.imagine_reset(start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data=True)
        # for _ in range(horizon):
        #     a = dreamer.act()
        #     dreamer.imagine_step(a, temperature=1)
        # z, o, _, a, r, g, d, weights = dreamer.get_data()

        # o = o[:, :-1, -1:]  # remove last time step and use last frame of frame stack
        # a, r, g, weights = [x.cpu().numpy() for x in (a, r, g, weights)]

        # imagine_img = o
        # if config.get('env_grayscale',True):
        #     imagine_img = imagine_img.unsqueeze(3)
        # else:
        #     imagine_img = imagine_img.permute(0, 1, 2, 5, 3, 4)
        # imagine_img = imagine_img.unsqueeze(1)
        # imagine_img = imagine_img.transpose(2, 3).flatten(0, 3)
        # pad = 2
        # extra_pad = 38
        # imagine_img = utils.make_grid(imagine_img, nrow=o.shape[1], padding=(pad + extra_pad, pad))
        # imagine_img = utils.to_image(imagine_img[:, extra_pad:])

        # draw = ImageDraw.Draw(imagine_img)
        # h, w = o.shape[3:5]
        # for t in range(r.shape[1]):
        #     for i in range(r.shape[0]):
        #         x = pad + t * (w + pad)
        #         y = pad + i * (h + extra_pad + pad) + h
        #         weight = weights[i, t]
        #         reward = r[i, t]

        #         if abs(reward) < 1e-4:
        #             color_rgb = int(weight * 255)
        #             color = (color_rgb, color_rgb, color_rgb)  # white
        #         elif reward > 0:
        #             color_rb = int(weight * 100)
        #             color_g = int(weight * (255 + reward * 255) / 2)
        #             color = (color_rb, color_g, color_rb)  # green
        #         else:
        #             color_gb = int(weight * 80)
        #             color_r = int(weight * (255 + (-reward) * 255) / 2)
        #             color = (color_r, color_gb, color_gb)  # red
        #         draw.text((x + 2, y + 2), f'a: {self.action_meanings[a[i, t]][:7]: >7}', fill=color)
        #         draw.text((x + 2, y + 2 + 10), f'r: {r[i, t]: .4f}', fill=color)
        #         draw.text((x + 2, y + 2 + 20), f'g: {g[i, t]: .4f}', fill=color)
        return recon_img, imagine_img

    @torch.no_grad()
    def _create_eval_images_atari(self, is_final=False):
        config = self.config
        agent = self.agent
        replay_buffer = self.replay_buffer
        obs_model = agent.wm.obs_model.eval()

        # recon_img
        idx = utils.random_choice(replay_buffer.size, 10, device=replay_buffer.device).unsqueeze(1)
        o = replay_buffer.get_obs(idx)
        z = obs_model.encode_sample(o, temperature=0)
        recons = obs_model.decode(z)
        # use last frame of frame stack
        o = o[:, :, -1:]
        recons = recons[:, :, -1:]
        if config.get('env_grayscale',True):
            recon_img = [o.unsqueeze(-3), recons.unsqueeze(-3)]  # unsqueeze channel
        else:
            recon_img = [o.permute(0, 1, 2, 5, 3, 4), recons.permute(0, 1, 2, 5, 3, 4)]
        recon_img = torch.cat(recon_img, dim=0).squeeze(1).transpose(0, 1).flatten(0, 1)
        recon_img = torchvision.utils.make_grid(recon_img, nrow=o.shape[0], padding=2)
        recon_img = utils.to_image(recon_img)

        # imagine_img
        idx = idx[:5]
        start_o = replay_buffer.get_obs(idx, prefix=1)  # 1 for context
        start_a = replay_buffer.get_actions(idx, prefix=1)[:, :-1]
        start_r = replay_buffer.get_rewards(idx, prefix=1)[:, :-1]
        start_terminated = replay_buffer.get_terminated(idx, prefix=1)[:, :-1]
        start_truncated = replay_buffer.get_truncated(idx, prefix=1)[:, :-1]
        start_z = obs_model.encode_sample(start_o, temperature=0)

        horizon = 100 if is_final else config['wm_sequence_length']
        dreamer = Dreamer(config, agent.wm, mode='imagine', ac=agent.ac, store_data=True,
                          start_z_sampler=self._create_start_z_sampler(temperature=0), always_compute_obs=True)
        dreamer.imagine_reset(start_z, start_a, start_r, start_terminated, start_truncated, keep_start_data=True)
        for _ in range(horizon):
            a = dreamer.act()
            dreamer.imagine_step(a, temperature=1)
        z, o, _, a, r, g, d, weights = dreamer.get_data()

        o = o[:, :-1, -1:]  # remove last time step and use last frame of frame stack
        a, r, g, weights = [x.cpu().numpy() for x in (a, r, g, weights)]

        imagine_img = o
        if config.get('env_grayscale',True):
            imagine_img = imagine_img.unsqueeze(3)
        else:
            imagine_img = imagine_img.permute(0, 1, 2, 5, 3, 4)
        imagine_img = imagine_img.unsqueeze(1)
        imagine_img = imagine_img.transpose(2, 3).flatten(0, 3)
        pad = 2
        extra_pad = 38
        imagine_img = utils.make_grid(imagine_img, nrow=o.shape[1], padding=(pad + extra_pad, pad))
        imagine_img = utils.to_image(imagine_img[:, extra_pad:])

        draw = ImageDraw.Draw(imagine_img)
        h, w = o.shape[3:5]
        for t in range(r.shape[1]):
            for i in range(r.shape[0]):
                x = pad + t * (w + pad)
                y = pad + i * (h + extra_pad + pad) + h
                weight = weights[i, t]
                reward = r[i, t]

                if abs(reward) < 1e-4:
                    color_rgb = int(weight * 255)
                    color = (color_rgb, color_rgb, color_rgb)  # white
                elif reward > 0:
                    color_rb = int(weight * 100)
                    color_g = int(weight * (255 + reward * 255) / 2)
                    color = (color_rb, color_g, color_rb)  # green
                else:
                    color_gb = int(weight * 80)
                    color_r = int(weight * (255 + (-reward) * 255) / 2)
                    color = (color_r, color_gb, color_gb)  # red
                draw.text((x + 2, y + 2), f'a: {self.action_meanings[a[i, t]][:7]: >7}', fill=color)
                draw.text((x + 2, y + 2 + 10), f'r: {r[i, t]: .4f}', fill=color)
                draw.text((x + 2, y + 2 + 20), f'g: {g[i, t]: .4f}', fill=color)
        return recon_img, imagine_img
