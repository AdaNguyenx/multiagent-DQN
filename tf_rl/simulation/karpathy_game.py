import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from collections import defaultdict, OrderedDict
from euclid import Circle, Point2, Vector2, LineSegment2
from copy import deepcopy

# from ..utils import svg
from IPython.display import clear_output, display, HTML

import cv2

class GameObject(object):
    def __init__(self, position, velocity, speed, obj_type, settings, name):
        """Esentially represents circles of different kinds, which have
        position and velocity."""
        self.settings = settings
        self.radius = self.settings["object_radius"]

        self.obj_type = obj_type
        self.position = position
        self.velocity  = velocity
        self.bounciness = 1.0
        self.speed = speed
        self.name = name


    def wall_collisions(self):
        """Update velocity upon collision with the wall."""
        
        world_size = self.settings["world_size"]

        for dim in range(2):
            if self.position[dim] - self.radius       <= 0               and self.velocity[dim] < 0:
                self.velocity[dim] = - self.velocity[dim] * self.bounciness
            elif self.position[dim] + self.radius + 1 >= world_size[dim] and self.velocity[dim] > 0:
                self.velocity[dim] = - self.velocity[dim] * self.bounciness
        

    def move(self, dt):
        """Move as if dt seconds passed"""
        self.position += dt * self.velocity
        self.position = Point2(*self.position)

    def step(self, dt):
        """Move and bounce of walls."""
        # only move if object is predator
        if self.obj_type == "pred":
            self.wall_collisions()
            # move in the direction of the previous action
            self.move(dt)

    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self, vis):
        """Return svg object for this item."""
        color = self.settings["colors"][self.obj_type]
        draw_position = (int(self.position[0] + 10), int(self.position[1] + 10))
        cv2.circle(vis, draw_position, int(self.radius), color, -1, cv2.LINE_AA)
        # if self.name != None:
        #     cv2.putText(vis, self.name, draw_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)


class KarpathyGame(object):
    def __init__(self, settings):
        """Initiallize game simulator with settings"""
        self.settings = settings
        self.size  = self.settings["world_size"]
        self.walls = [LineSegment2(Point2(0,0),                        Point2(0,self.size[1])),
                      LineSegment2(Point2(0,self.size[1]),             Point2(self.size[0], self.size[1])),
                      LineSegment2(Point2(self.size[0], self.size[1]), Point2(self.size[0], 0)),
                      LineSegment2(Point2(self.size[0], 0),            Point2(0,0))]


        #number of [prey, pred] controlling movement
        self.num_active = []
        if self.settings["network_prey"] == 'one' and self.settings["network_pred"] == 'one':
            if self.settings['num_objects_active']['prey'] != 0:    
                self.num_active.append(self.settings['num_objects_active']['prey'])
            if self.settings['num_objects_active']['pred'] != 0:
                self.num_active.append(self.settings['num_objects_active']['pred'])
        elif self.settings["network_prey"] == 'one' and self.settings["network_pred"] == 'multiple':
            if self.settings['num_objects_active']['prey'] != 0:    
                self.num_active.append(self.settings['num_objects_active']['prey'])
            for i in range(self.settings['num_objects_active']['pred']):
                self.num_active.append(1)
        elif self.settings["network_prey"] == 'multiple' and self.settings["network_pred"] == 'one':
            for i in range(self.settings['num_objects_active']['prey']):
                self.num_active.append(1)
            if self.settings['num_objects_active']['pred'] != 0:
                self.num_active.append(self.settings['num_objects_active']['pred'])
        else:
            for i in range(self.settings['num_objects_active']['prey']):
                self.num_active.append(1)
            for i in range(self.settings['num_objects_active']['pred']):
                self.num_active.append(1)

        self.objects = []
        self.num_objects_init = deepcopy(self.settings["num_objects"])

        self.cues = self.settings["cue_types"] # left
        self.current_cue = 0

        self.reset()

        self.observation_lines = self.generate_observation_lines()

        self.object_reward = np.zeros(sum(self.num_active))
        self.collected_rewards = []
        self.done = False

        # every observation_line sees one of objects or wall 
        self.eye_observation_size = len(self.settings["objects"]) + 1
        # additionally there are two numbers representing agents own velocity and position.
        # self.observation_size = self.eye_observation_size * len(self.observation_lines) + 2 + 2
        self.observation_size = self.eye_observation_size * len(self.observation_lines) 

        #Four possible movement directions plus stationary
        self.directions = [Vector2(*d) for d in [[1,0], [0,1], [-1,0],[0,-1],[0.0,0.0]]]

        self.num_actions = len(self.directions)

        preys = []
        for i in range(self.settings['num_objects_active']['prey']):
            name = 'pred' + str(i)
            preds.append(name)
        preds = []
        for i in range(self.settings['num_objects_active']['pred']):
            name = 'pred' + str(i)
            preds.append(name)
        if preys == []:
            self.objects_eaten = OrderedDict((pred, 0) for pred in preds)
        elif preds == []:
            self.objects_eaten = OrderedDict((prey, 0) for prey in preys)
        else:
            self.objects_eaten = OrderedDict(((prey, 0) for prey in preys), ((pred, 0) for pred in preds))
        
        self.num_acts_so_far = 0
        self.list_to_remove = []
        self.collisions = np.zeros(5)

        tests = float(self.settings["maximum_velocity"]['prey'])

        self.max_velocity = max(float(self.settings["maximum_velocity"]['prey']),
                                    float(self.settings["maximum_velocity"]['pred']))

    def spawn(self, obj_type, number):
        radius = self.settings["object_radius"]
        # prey
        if obj_type == "prey":
            self.current_cue = np.random.choice(self.cues)
            cue_point = [250, 200]
            radius  = 150
            angle = np.linspace(0, -np.pi, self.cues)[self.current_cue]
            rotation = [math.cos(angle), math.sin(angle)]
            position_center = [radius * rotation[0] + cue_point[0], radius * rotation[1] + cue_point[1]]
            for _ in range(number):
                position = np.random.uniform([position_center[0] - 25, position_center[1] - 25], [position_center[0] + 25, position_center[1] + 25])
                position = Point2(float(position[0]), float(position[1]))
                self.objects.append(GameObject(position, Vector2(0.0, 0.0), 0.0, obj_type, self.settings, None))

        # predator
        elif obj_type == "pred":
            max_velocity = float(self.settings["maximum_velocity"][obj_type])
            for i in range(number):
                velocity = np.random.uniform(-max_velocity, max_velocity, 2).astype(float)
                velocity = Vector2(float(velocity[0]), float(velocity[1]))
                position = np.random.uniform([225, 225], [275, 275])
                position = Point2(float(position[0]), float(position[1]))
                name = str(i)
                self.objects.append(GameObject(position, velocity, max_velocity, obj_type, self.settings, name))
        # cue
        else:
            position = [250, 200]
            position = Point2(float(position[0]), float(position[1]))
            velocity = self.current_cue + 1
            self.objects.append(GameObject(position, velocity, 0.0, obj_type, self.settings,None))


    def reset(self):
        self.settings["num_objects"] = deepcopy(self.num_objects_init)
        if self.objects != []:
            self.objects = []
            self.done = False
        for obj_type, number in self.num_objects_init.items():
            self.spawn(obj_type, number)

    def perform_action(self, action_id, obj_type):
        """Change velocity to one of the individual's vectors"""
        for ind, active_ind in enumerate(self.get_list(obj_type)):
            assert 0 <= action_id[ind] < self.num_actions
            self.objects[active_ind].velocity *= 0.5
            self.objects[active_ind].velocity += (
            	self.directions[int(action_id[ind])] * self.settings["delta_v"])

    def spawn_object(self, obj_type):
        """Spawn object of a given type and add it to the objects array"""
        radius = self.settings["object_radius"]
        position = np.random.uniform([radius, radius], np.array(self.size) - radius)
        position = Point2(float(position[0]), float(position[1]))
        max_velocity = float(self.settings["maximum_velocity"][obj_type])
        velocity = np.random.uniform(-max_velocity, max_velocity, 2).astype(float)
        velocity = Vector2(float(velocity[0]), float(velocity[1]))
        return GameObject(position, velocity, max_velocity, obj_type, self.settings, None) 


    def step(self, dt, num_types):
        """Simulate all the objects for a given ammount of time.
           and resolve collisions with other objects"""
        for obj in self.objects:
            obj.step(dt)
        # num_types are only active objects
        for i in range(num_types):
            self.resolve_collisions(i)
        #remove objects
        for i in self.list_to_remove:
            # respawn only if the objects are predators
            if self.objects[i].obj_type == "pred":
                if self.settings["collision_penalty"]:
                    self.objects[i] = self.spawn_object(self.objects[i].obj_type) # for respawn after collision
            elif self.objects[i].obj_type == "prey":
                self.objects.remove(self.objects[i])
                self.settings['num_objects']['prey'] -= 1
                if self.settings['num_objects']['prey'] == 0:
                    self.done = True
        self.list_to_remove = []


    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def get_list(self, obj_type):
        if self.settings['num_objects_active']['prey'] == 0:
            return(self.get_active_objects("pred", obj_type))
        elif self.settings['num_objects_active']['pred'] == 0:
            return(self.get_active_objects("prey", obj_type))
        else:
            if (self.settings['network_prey'] == 'one' and obj_type == 0) or (self.settings['network_prey'] == 'multiple' and obj_type < self.settings['num_objects_active']['prey']):
                return(self.get_active_objects("prey", obj_type))
            else:
                return(self.get_active_objects("pred", obj_type))

    def get_active_objects(self, type, obj_type):
        if type == "prey":
            return(range(obj_type, self.num_active[obj_type] + obj_type))
        elif type == "pred":
            return(range(self.settings['num_objects']['prey'] + obj_type, self.settings['num_objects']['prey'] + self.num_active[obj_type] + obj_type))


    def resolve_collisions(self, obj_type):
        """If hero touches, hero eats. Also reward gets updated."""
        #assumes all individuals are the same size
        collision_distance = 2 * self.settings["object_radius"]
        collision_distance2 = collision_distance ** 2
        for ind, active_ind in enumerate(self.get_list(obj_type)):
 
            to_remove = []

            for i, obj in enumerate(self.objects): 
                if i == active_ind:
                    continue 
                if self.squared_distance(self.objects[active_ind].position, obj.position) < collision_distance2:
                    to_remove.append(i)
            # for prey
            if self.objects[active_ind].obj_type == "prey":
                for i in to_remove:
                    self.object_reward[sum(self.num_active[:obj_type]) + ind] += self.settings["object_reward"][self.objects[active_ind].obj_type][self.objects[i].obj_type]
                    # comment out the next 4 lines to prevent preys from eating each other
                    if (self.objects[i].obj_type == "prey"): 
                        self.list_to_remove.append(i)
                        name = "pred" + str(obj_type)
                        self.objects_eaten[name] +=1   
                        
            # for predator
            else:
                for i in to_remove:
                    self.object_reward[sum(self.num_active[:obj_type]) + ind] += self.settings["object_reward"][self.objects[active_ind].obj_type][self.objects[i].obj_type]
                    if (self.objects[i].obj_type == "pred"):
                        if self.settings["collision_penalty"]:
                            self.list_to_remove.append(i)  #comment out this line to allow predators to hit each other
                        self.collisions[obj_type] += 1
                        continue
                    elif (self.objects[i].obj_type == "prey"):
                        self.list_to_remove.append(i)
                        name = "pred" + str(obj_type)
                        self.objects_eaten[name] +=1
                    else:
                        continue

    def inside_walls(self, point):
        """Check if the point is inside the walls"""
        EPS = 1e-4
        return (EPS <= point[0] < self.size[0] - EPS and
                EPS <= point[1] < self.size[1] - EPS)

    def observe(self, obj_type, certainty):
        """Return observation vector. For all the observation directions it returns representation
        of the closest object to the hero - might be nothing, another object or a wall.
        Representation of observation for all the directions will be concatenated.
        """
        # For each active agent, observation is a vector 
        num_obj_types = len(self.settings["objects"]) + 1 # and wall
        #max_velocity_x, max_velocity_y = self.settings["maximum_velocity"]

        observable_distance = self.settings["observation_line_length"]

        observation = np.zeros((self.num_active[obj_type], self.observation_size), dtype = float)

        
        for ind, active_ind in enumerate(self.get_list(obj_type)):
    
            relevant_objects = [obj for obj in self.objects[:active_ind] + self.objects[active_ind + 1:] 
                                if obj.position.distance(self.objects[active_ind].position) < observable_distance]

            # objects sorted from closest to furthest
            relevant_objects.sort(key=lambda x: x.position.distance(self.objects[active_ind].position))

            observation_offset = 0
            for i, observation_line in enumerate(self.generate_observation_lines()):
                # shift to hero position
                observation_line = LineSegment2(self.objects[active_ind].position + Vector2(*observation_line.p1),
                                                self.objects[active_ind].position + Vector2(*observation_line.p2))
                observed_object = None
                # if end of observation line is outside of walls, we see the wall.
                if not self.inside_walls(observation_line.p2):
                    observed_object = "**wall**"
                for obj in relevant_objects:
                    if observation_line.distance(obj.position) < self.settings["object_radius"]:
                        observed_object = obj
                        break
                object_type_id = None

                # wall seen
                if observed_object == "**wall**": 
                    object_type_id = num_obj_types - 1

                # agent seen
                elif observed_object is not None: 
                    object_type_id = self.settings["objects"].index(observed_object.obj_type)

                if object_type_id is not None: 
                    observation[ind, observation_offset + object_type_id] = 1.0
                    # for cue
                    if observed_object != "**wall**":
                        if observed_object.obj_type == "cue":
                            # uncertainty for cue
                            if random.random() < certainty:
                                observation[ind, observation_offset + object_type_id] = observed_object.velocity
                            else:
                                observation[ind, observation_offset + object_type_id] = np.random.choice(self.cues) + 1

                observation_offset += self.eye_observation_size

        return observation

    def distance_to_walls(self, obj):
        """Returns distance of a hero to walls"""
        res = float('inf')
        for wall in self.walls:
            res = min(res, obj.position.distance(wall))
        return res - self.settings["object_radius"]

    def collect_reward(self, obj_type):
        """Return accumulated object eating score + current distance to walls score"""
        
        total_reward = np.zeros(self.num_active[obj_type])

        for ind, active_ind in enumerate(self.get_list(obj_type)):

            if self.settings["wall_distance_penalty"] != 0: 
                wall_reward =  (self.settings["wall_distance_penalty"] * 
                               np.exp(-self.distance_to_walls(self.objects[active_ind]) / self.settings["tolerable_distance_to_wall"]))
                assert wall_reward < 1e-3, "You are rewarding hero for being close to the wall!"
            else:
                wall_reward = 0
            
            total_reward[ind] = wall_reward + self.object_reward[sum(self.num_active[:obj_type]) + ind]
            self.object_reward[sum(self.num_active[:obj_type]) + ind] = 0 # reset reward
            self.collected_rewards.append(total_reward[ind])
        return total_reward

    def update_num_actions(self, num_acts):
        self.num_acts_so_far = num_acts

    def plot_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.collected_rewards[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))
        plt.plot(list(range(len(x))), x)

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        result = []

        start = Point2(0.0, 0.0)
        end   = Point2(self.settings["observation_line_length"],
                       self.settings["observation_line_length"])

        # for angle in np.linspace(5*np.pi/4, 7*np.pi/4, self.settings["num_observation_lines"], endpoint=False):
        for angle in np.linspace(0, 2*np.pi, self.settings["num_observation_lines"], endpoint=False):    
            rotation = Point2(math.cos(angle), math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append( LineSegment2(current_start, current_end))
        return result

    def setup_draw(self):
        """
        An optional method to be triggered in simulate(...) to initialise
        the figure handles for rendering.
        simulate(...) will run with/without this method declared in the simulation class
        As we are using SVG strings in KarpathyGame, it is not curently used.
        """
        pass

    def draw(self, stats=[]):
        """
        An optional method to be triggered in simulate(...) to render the simulated environment.
        It is repeatedly called in each simulated iteration.
        simulate(...) will run with/without this method declared in the simulation class.
        """

        stats = stats[:]
        recent_reward = self.collected_rewards[-500:] + [0]
        objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in self.objects_eaten.items()])
        stats.extend([
            #"nearest wall = %.1f" % (self.distance_to_walls(self.objects[0]),),
            "reward = %.7f" % (sum(recent_reward)/float(len(recent_reward))*100.0,),
            "Objects Eaten => %s" % (objects_eaten_str,),
            "Number of actions so far => %.1f" % (self.num_acts_so_far,),

        ])

        visualisation = np.zeros((self.size[1] + 20 + 20 * len(stats), self.size[0] + 20, 3), np.uint8) 
        visualisation[:,:,0] = visualisation[:,:,0] + 160 
        visualisation[:,:,1] = visualisation[:,:,1] + 124
        visualisation[:,:,2] = visualisation[:,:,2] + 110
        

        cv2.rectangle(visualisation, (10, 10), (self.size[0]+10, self.size[1]+10), [255,255,255])

        
        # Draw observation lines
        for obj in self.objects:
            obj.draw(visualisation)
            # if obj.obj_type == "pred":
            #     for line in self.observation_lines:
            #         point1 = obj.position + line.p1
            #         point1 = tuple((int(point1[0] + 10), int(point1[1] + 10)))
            #         point2 = obj.position + line.p2
            #         point2 = tuple((int(point2[0] + 10), int(point2[1] + 10)))
                    

        offset = self.size[1] + 15
        for txt in stats:
            cv2.putText(visualisation, txt, (10, offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        [84, 37, 0], lineType=cv2.LINE_AA)
            offset += 20

        cv2.imshow('rl_schooling', visualisation)
        cv2.waitKey(1)

    def shut_down_graphics(self):
        for indx in range(20):
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
