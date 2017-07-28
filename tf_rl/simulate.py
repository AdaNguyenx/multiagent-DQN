from __future__ import division

import math
import time

import matplotlib.pyplot as plt
from itertools import count
from os.path import join, exists
from os import makedirs
from IPython.display import clear_output, display, HTML

import numpy as np
import tensorflow as tf 
import random

from baselines.common.schedules import LinearSchedule

def simulate(simulation,
             replay = None,
             act = None,
             train = None,
             update = None,
             debug = None,
             # act= None,
             fps=60,
             visualize_every=1,
             action_every=1,
             simulation_resolution=None,
             wait=False,
             disable_training=False,
             save_path=None,
             timesteps=None,
             elapsed = [],
             all_rewards = [],
             percent = None,
             certainty = 0.0,
             draw = 0):
    """Start the simulation. Performs three tasks

        - visualizes simulation
        - advances simulator state
        - reports state to act and chooses actions
          to be performed.

    Parameters
    -------
    simulation: tr_lr.simulation
        simulation that will be simulated ;-)
    act: tr_lr.act
        act used
    fps: int
        frames per seconds
    visualize_every: int
        visualize every `visualize_every`-th frame.
    action_every: int
        take action every `action_every`-th frame
    simulation_resolution: float
        simulate at most 'simulation_resolution' seconds at a time.
        If None, the it is set to 1/FPS (default).
    wait: boolean
        whether to intentionally slow down the simulation
        to appear real time.
    disable_training: bool
        if true training_step is never called.
    save_path: str
        save svg visualization (only tl_rl.utils.svg
        supported for the moment)
    """

    # prepare path to save simulation images
    if save_path is not None:
        if not exists(save_path):
            makedirs(save_path)
    last_image = 0

    # calculate simulation times
    chunks_per_frame = 1
    chunk_length_s   = 1.0 / fps

    if simulation_resolution is not None:
        frame_length_s = 1.0 / fps
        chunks_per_frame = int(math.ceil(frame_length_s / simulation_resolution))
        chunks_per_frame = max(chunks_per_frame, 1)
        chunk_length_s = frame_length_s / chunks_per_frame

    # state transition bookkeeping
    last_observation = [None] * len(act) 
    last_action     = [None] * len(act)

    simulation_started_time = time.time()

    observation_size = simulation.observation_size

    # start observation
    for i in range(len(act)):
        last_observation[i] = simulation.observe(i, certainty)

    num_actions = 0
    each_num_actions = np.zeros(len(act))
    exploration = LinearSchedule(schedule_timesteps=5000, initial_p=1.0, final_p=0.02)

    trials = 0
    start = time.time()

    for frame_no in count():

        timesteps[0] = frame_no 

        if trials == 100:
            print("done")
            break 

        # monitoring
        if frame_no % 5000 == 0:
            print(trials)

        for _ in range(chunks_per_frame):
            #move all the individuals for one time step
            #deal with any collisions
            simulation.step(chunk_length_s, len(act))

        # frame skipping
        if frame_no % action_every == 0:
            new_observation = [None] * len(act)
            reward          = [None] * len(act) 
            new_action = [None] * len(act)

            if simulation.done:
                trials += 1
                elapsed.append(time.time() - start)
                total = []
                for o,r in simulation.objects_eaten.items():
                    total.append(r)
                all_rewards.append(total)
                start = time.time()
                done = 1
                simulation.reset()
            else:
                done = 0

            # produce action from network, then pool for majority
            for i in range(len(act)):
                new_action[i] = act[i](last_observation[i], update_eps=exploration.value(each_num_actions[i]))[0]

            action_freq = np.bincount(new_action)
            consensus = np.argwhere(action_freq == max(action_freq))
            if len(consensus) > 1:
                consensus = np.random.choice(len(consensus))
            
            #go through each network group 
            for i in range(len(act)):

                if percent is not None:
                    if random.random() < percent:
                        new_action[i] = [new_action[i]]
                    else:
                        new_action[i] = [consensus]
                else:
                    new_action[i] = [new_action[i]]

                new_action[i] = act[i](last_observation[i], update_eps=0.02)

                simulation.perform_action(new_action[i], i)

                each_num_actions[i] += 1
                num_actions += 1
                simulation.update_num_actions(num_actions)

                new_observation[i] = simulation.observe(i, certainty)
                
                reward[i] = simulation.collect_reward(i)

                # store last transition
                for j in range(len(reward[i])):
                    replay[i].add(last_observation[i][j], new_action[i][j], reward[i][j], new_observation[i][j], done)

                # update current state as last state.
                last_action[i] = new_action[i]
                last_observation[i] = new_observation[i]

                # train
                if frame_no > 100:
                    obses_t, actions, rewards, obses_tp1, dones = replay[i].sample(32)
                    obses_t = np.reshape(obses_t, (-1, observation_size))
                    obses_tp1 = np.reshape(obses_tp1, (-1, observation_size))
                    rewards = np.reshape(rewards, (-1,))
                    actions = np.reshape(actions, (-1,))
                    train[i](obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                # update target network periodically
                if frame_no % 1000 == 0:
                    update[i]()

        # adding 1 to make it less likely to happen at the same time as
        # action taking.
        if draw:
            if (frame_no + 1) % visualize_every == 0:
                fps_estimate = frame_no / (time.time() - simulation_started_time)

                # draw simulated environment all the rendering is handled within the simulation object 
                stats = ["fps = %.1f" % (fps_estimate, )]
                if hasattr(simulation, 'draw'): # render with the draw function
                    simulation.draw(stats) 
                elif hasattr(simulation, 'to_html'): # in case some class only support svg rendering
                    clear_output(wait=True)
                    svg_html = simulation.to_html(stats)
                    display(svg_html)

                if save_path is not None:
                    img_path = join(save_path, "%d.svg" % (last_image,))
                    with open(img_path, "w") as f:
                        svg_html.write_svg(f)
                    last_image += 1

            time_should_have_passed = frame_no / fps
            time_passed = (time.time() - simulation_started_time)
            if wait and (time_should_have_passed > time_passed):
                time.sleep(time_should_have_passed - time_passed)


