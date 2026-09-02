"""Connectivity-by-construction TRACE environment without ML-framework code."""
from __future__ import annotations

from collections import deque

import numpy as np
import scipy.ndimage

from biotsavart import build_coupling
from gradient_topopt import R_ROI
from physics import THETA_OFFSET, WIRING


ACTION_NAMES = ("left", "right", "up", "down")
ACTION_DELTAS = np.asarray(((-1, 0), (1, 0), (0, 1), (0, -1)), dtype=int)
TARGET_GRADIENT = 0.01


class SeqTOEnvironment:
    def __init__(self, axis, problem, half_width=0, volume_fraction=0.20,
                 max_steps=None, observation_shape=(32, 20), history=4,
                 boundary_penalty=-2.0, max_invalid_actions=24):
        self.axis = axis.lower()
        self.problem = problem
        self.nl, self.nz = problem.mesh.nl, problem.mesh.nz
        self.half_width = int(half_width)
        self.volume_fraction = float(volume_fraction)
        self.material_budget = int(round(self.volume_fraction*self.nl*self.nz))
        self.max_steps = int(max_steps or 3*self.nl)
        self.observation_shape = tuple(observation_shape)
        self.history_length = int(history)
        self.boundary_penalty = float(boundary_penalty)
        self.max_invalid_actions = int(max_invalid_actions)
        self.i_min = self.half_width
        self.i_max = self.nl-1-self.half_width
        self.j_min = self.half_width
        self.j_max = self.nz-1-self.half_width
        self.start = (self.i_min, self.j_max)
        self.goal_i = self.i_max
        self._build_gradient_operator()

    def _build_gradient_operator(self, roi_radius_mm=None):
        radius = 0.5*R_ROI if roi_radius_mm is None else float(roi_radius_mm)*1e-3
        coordinate = np.linspace(-radius, radius, 9)
        xx, yy, zz = np.meshgrid(coordinate, coordinate, coordinate,
                                 indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        points = points[np.linalg.norm(points, axis=1) <= radius]
        step = 1e-4
        shift = np.eye(3)["xyz".index(self.axis)]*step
        kwargs = {"wiring": WIRING[self.axis]}
        if self.axis == "x":
            kwargs["theta_offset"] = THETA_OFFSET[self.axis]
        plus = build_coupling(self.problem.mesh, points+shift, **kwargs)
        minus = build_coupling(self.problem.mesh, points-shift, **kwargs)
        self.gradient_operator = (plus-minus)/(2*step)

    def reset(self):
        self.mask = np.zeros((self.nl, self.nz), dtype=np.float32)
        self.position = self.start
        self.steps = 0
        self.invalid_actions = 0
        self.reached_goal = False
        self._paint(*self.position)
        self.history = deque(
            [self.mask.copy() for _ in range(self.history_length)],
            maxlen=self.history_length)
        return self.observation()

    def _paint(self, i, j):
        h = self.half_width
        self.mask[max(0, i-h):min(self.nl, i+h+1),
                  max(0, j-h):min(self.nz, j+h+1)] = 1.0

    def _inside(self, i, j):
        return self.i_min <= i <= self.i_max and self.j_min <= j <= self.j_max

    def _resize_channel(self, channel):
        zoom = (self.observation_shape[0]/self.nl,
                self.observation_shape[1]/self.nz)
        return scipy.ndimage.zoom(channel, zoom, order=0, mode="nearest",
                                  prefilter=False)

    def observation(self):
        channels = [self._resize_channel(frame) for frame in self.history]
        controller = np.zeros(self.observation_shape, dtype=np.float32)
        oi = min(self.observation_shape[0]-1, int(round(
            self.position[0]*(self.observation_shape[0]-1)/max(1, self.nl-1))))
        oj = min(self.observation_shape[1]-1, int(round(
            self.position[1]*(self.observation_shape[1]-1)/max(1, self.nz-1))))
        controller[oi, oj] = 1.0
        target = np.zeros(self.observation_shape, dtype=np.float32)
        goal_oi = min(self.observation_shape[0]-1, int(round(
            self.goal_i*(self.observation_shape[0]-1)/max(1, self.nl-1))))
        target[goal_oi, :] = 1.0
        channels.extend((controller, target))
        return np.stack(channels, axis=-1).astype(np.float32)

    def _field_metrics(self):
        _, _, fem = self.problem.evaluate(self.mask.ravel(), want_grad=False)
        current_density = fem["Jl"]*fem["Vin"]
        gradient = self.gradient_operator @ current_density
        error = 100*np.max(np.abs(TARGET_GRADIENT-gradient)/TARGET_GRADIENT)
        return {
            "error_percent": float(error),
            "resistance_ohm": float(fem["R"]),
            "current_A": abs(float(fem["current"])),
            "power_W": float(fem["current"]**2*fem["R"]),
        }

    def connected_components(self):
        return int(scipy.ndimage.label(self.mask > 0.5)[1])


class StabilizedSeqTOEnvironment(SeqTOEnvironment):
    """Four-neighbour TRACE environment with a physical-width copper brush."""
    def __init__(self, axis, problem, wire_width_mm=1.0,
                 volume_fraction=0.20, max_steps=None,
                 observation_shape=(32, 20), history=4,
                 boundary_penalty=-2.0, max_invalid_actions=24,
                 progress_weight=0.03, material_penalty_weight=10.0,
                 quality_bonus_weight=12.0, resistance_weight=0.0,
                 resistance_limit_ohm=None, goal_row_fraction=None,
                 exact_state=False, roi_radius_mm=None):
        super().__init__(axis, problem, half_width=0,
                         volume_fraction=volume_fraction,
                         max_steps=max_steps or 3*problem.mesh.nl,
                         observation_shape=observation_shape, history=history,
                         boundary_penalty=boundary_penalty,
                         max_invalid_actions=max_invalid_actions)
        if roi_radius_mm is not None:
            self._build_gradient_operator(roi_radius_mm)
        self.wire_width_m = float(wire_width_mm)*1e-3
        if self.wire_width_m <= 0:
            raise ValueError("wire_width_mm must be positive")
        self.progress_weight = float(progress_weight)
        self.material_penalty_weight = float(material_penalty_weight)
        self.quality_bonus_weight = float(quality_bonus_weight)
        self.resistance_weight = float(resistance_weight)
        self.resistance_limit_ohm = (None if resistance_limit_ohm in (None, 0, 0.0)
                                     else float(resistance_limit_ohm))
        self.goal_row_fraction = goal_row_fraction
        self.exact_state = bool(exact_state)
        self.brush_offsets = self._physical_brush_offsets()
        self.i_radius = int(np.max(np.abs(self.brush_offsets[:, 0])))
        self.j_radius = int(np.max(np.abs(self.brush_offsets[:, 1])))
        self.i_min, self.i_max = self.i_radius, self.nl-1-self.i_radius
        self.j_min, self.j_max = self.j_radius, self.nz-1-self.j_radius
        self.start = (self.i_min, self.j_max)
        self.goal_i = self.i_max
        self.goal_j = None if goal_row_fraction is None else int(round(
            self.j_min+float(goal_row_fraction)*(self.j_max-self.j_min)))
        self.previous_position = None

    def _physical_brush_offsets(self):
        radius = 0.5*self.wire_width_m
        ri = max(0, int(np.floor(radius/self.problem.mesh.dl)))
        rj = max(0, int(np.floor(radius/self.problem.mesh.dz)))
        offsets = [(di, dj) for di in range(-ri, ri+1)
                   for dj in range(-rj, rj+1)
                   if np.hypot(di*self.problem.mesh.dl,
                               dj*self.problem.mesh.dz) <= radius+1e-15]
        if (0, 0) not in offsets:
            offsets.append((0, 0))
        return np.asarray(offsets, dtype=int)

    def _paint(self, i, j):
        # During the base constructor the physical brush has not yet been set.
        if not hasattr(self, "brush_offsets"):
            return super()._paint(i, j)
        ii = i+self.brush_offsets[:, 0]
        jj = j+self.brush_offsets[:, 1]
        valid = (0 <= ii) & (ii < self.nl) & (0 <= jj) & (jj < self.nz)
        self.mask[ii[valid], jj[valid]] = 1.0

    def reset(self):
        self.previous_position = None
        return super().reset()

    def observation(self):
        observation = super().observation()
        if self.goal_j is not None:
            target = np.zeros(self.observation_shape, dtype=np.float32)
            oi = int(round(self.goal_i*(self.observation_shape[0]-1)/
                           max(1, self.nl-1)))
            oj = int(round(self.goal_j*(self.observation_shape[1]-1)/
                           max(1, self.nz-1)))
            target[oi, oj] = 1.0
            observation[..., -1] = target
        if self.exact_state:
            previous_di = previous_dj = 0.0
            if self.previous_position is not None:
                previous_di = self.position[0]-self.previous_position[0]
                previous_dj = self.position[1]-self.previous_position[1]
            scalars = (self.position[0]/max(1, self.nl-1),
                       self.position[1]/max(1, self.nz-1),
                       previous_di, previous_dj,
                       self.steps/max(1, self.max_steps),
                       self.mask.sum()/max(1, self.material_budget))
            planes = np.stack([np.full(self.observation_shape, value,
                                       dtype=np.float32)
                               for value in scalars], axis=-1)
            observation = np.concatenate((observation, planes), axis=-1)
        return observation

    def reached_return_port(self):
        if self.position[0] < self.goal_i:
            return False
        return self.goal_j is None or abs(self.position[1]-self.goal_j) <= self.j_radius

    def admissible_action_mask(self):
        mask = np.ones(4, dtype=bool)
        for action, (di, dj) in enumerate(ACTION_DELTAS):
            candidate = (self.position[0]+int(di), self.position[1]+int(dj))
            if not self._inside(*candidate):
                mask[action] = False
            if self.previous_position is not None and candidate == self.previous_position:
                mask[action] = False
        if not mask.any():
            for action, (di, dj) in enumerate(ACTION_DELTAS):
                candidate = (self.position[0]+int(di), self.position[1]+int(dj))
                mask[action] = self._inside(*candidate)
        return mask

    def step(self, action, physics_weight=1.0):
        action = int(action)
        if not 0 <= action < 4:
            raise ValueError("action must be 0:left, 1:right, 2:up, or 3:down")
        old_position = self.position
        old_i, old_j = old_position
        di, dj = ACTION_DELTAS[action]
        new_i, new_j = old_i+int(di), old_j+int(dj)
        self.steps += 1
        reward = -0.002
        invalid = not self._inside(new_i, new_j)
        old_material = int(self.mask.sum())
        if invalid:
            self.invalid_actions += 1
            reward += self.boundary_penalty
            new_i, new_j = old_i, old_j
        else:
            self.previous_position = old_position
            self.position = (new_i, new_j)
            self._paint(new_i, new_j)
            effective = self.progress_weight*(1.0-0.8*physics_weight)
            reward += effective*((self.goal_i-old_i)-(self.goal_i-new_i))
            if int(self.mask.sum()) == old_material:
                reward -= 0.03
        self.history.append(self.mask.copy())
        self.reached_goal = self.reached_return_port()
        done = (self.reached_goal or self.steps >= self.max_steps or
                self.invalid_actions >= self.max_invalid_actions)
        occupied = int(self.mask.sum())
        info = {"invalid_action": bool(invalid),
                "invalid_actions": self.invalid_actions,
                "occupied": occupied,
                "material_budget": self.material_budget,
                "reached_goal": bool(self.reached_goal),
                "components": self.connected_components(),
                "volume_fraction": occupied/(self.nl*self.nz)}
        if done:
            if self.reached_goal:
                reward += 40.0
                excess = max(0.0, occupied/max(1, self.material_budget)-1.0)
                reward -= self.material_penalty_weight*excess*excess
                metrics = self._field_metrics()
                clipped = min(metrics["error_percent"], 100.0)
                reward += float(physics_weight)*(
                    self.quality_bonus_weight*np.exp(-clipped/5.0)-clipped)
                info.update(metrics)
                if self.resistance_weight > 0 and self.resistance_limit_ohm:
                    excess_r = max(0.0, metrics["resistance_ohm"]/
                                   self.resistance_limit_ohm-1.0)
                    reward -= float(physics_weight)*self.resistance_weight*excess_r
                info["valid_topology"] = True
            else:
                remaining = (self.goal_i-self.position[0])/max(
                    1, self.goal_i-self.i_min)
                reward += -40.0-10.0*max(0.0, remaining)
                info["valid_topology"] = False
        return self.observation(), float(reward), bool(done), info


def curriculum_weights(episode, pretrain_episodes, physics_ramp_episodes,
                       entropy_start, entropy_end, total_episodes):
    if episode <= pretrain_episodes:
        physics = 0.0
    else:
        physics = min(1.0, (episode-pretrain_episodes)/
                      max(1, physics_ramp_episodes))
    fraction = (episode-1)/max(1, total_episodes-1)
    entropy = entropy_start*(entropy_end/entropy_start)**fraction
    return float(physics), float(entropy)


def clone_trajectory(trajectory):
    return {key: ([np.asarray(v).copy() for v in value]
                  if key in ("observations", "action_masks") else list(value))
            for key, value in trajectory.items()}

