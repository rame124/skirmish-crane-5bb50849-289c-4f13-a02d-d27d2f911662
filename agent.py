from enum import Enum, auto

from sandbox.crane import action, me, paths, tile, units, visible
from sandbox.observation_types import (
    AxialPosition,
    SkirmishAction,
    SkirmishObservation,
)


class State(Enum):
    ASSEMBLE = auto()
    ADVANCE = auto()
    SCOUT = auto()
    REGROUP = auto()
    ENGAGE = auto()


class Agent:
    """
    Season 2 terrain-aware finite state machine.

    Strategy:
    - Footman and archer assemble on advantageous terrain.
    - Archer seeks hills/forest when protected by allies.
    - Footman pushes enemies while the archer follows.
    - Cavalry scouts after the group assembles.
    - Cavalry regroups when unsupported.
    - Cavalry regroups at half health instead of fighting alone.
    - Once together, all units attack using the same target rule.
    """

    def reset(self, seed, observation) -> None:
        self.state = State.ASSEMBLE

    def act(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        self._update_state(observation)

        unit_type = me.unit_type(observation)

        if unit_type == "archer":
            return self._act_archer(observation)

        if unit_type == "footman":
            return self._act_footman(observation)

        if unit_type == "cavalry":
            return self._act_cavalry(observation)

        return action.stay()

    # ============================================================
    # STATE MACHINE
    # ============================================================

    def _update_state(
        self,
        observation: SkirmishObservation,
    ) -> None:

        unit_type = me.unit_type(observation)
        enemies = visible.enemies(observation)

        # --------------------------------------------------------
        # CAVALRY STATE MACHINE
        # --------------------------------------------------------

        if unit_type == "cavalry":

            hp = me.hit_points(observation)
            max_hp = units.STATS["cavalry"].hit_points

            allies = visible.allies(observation)

            grouped = self._has_nearby_ally(
                observation,
                distance=3,
            )

            # Half health means return to the team.
            if hp <= max_hp / 2:

                if grouped and enemies:
                    self.state = State.ENGAGE
                else:
                    self.state = State.REGROUP

                return

            # Enemy contact.
            if enemies:

                target = self._choose_target(
                    observation,
                    enemies,
                )

                supported = any(
                    self._ally_can_support(
                        ally,
                        target,
                    )
                    for ally in allies
                )

                if supported or grouped:
                    self.state = State.ENGAGE
                else:
                    self.state = State.REGROUP

                return

            # No enemies visible.

            if self.state == State.ENGAGE:
                self.state = State.SCOUT
                return

            if self.state == State.REGROUP:

                if grouped:
                    self.state = State.ADVANCE

                return

            if self.state == State.ASSEMBLE:

                footman = self._find_ally_type(
                    observation,
                    "footman",
                )

                archer = self._find_ally_type(
                    observation,
                    "archer",
                )

                if (
                    footman is not None
                    and archer is not None
                    and tile.distance(
                        footman["position"],
                        archer["position"],
                    ) <= 3
                ):
                    self.state = State.SCOUT

                return

            return

        # --------------------------------------------------------
        # ARCHER / FOOTMAN STATE MACHINE
        # --------------------------------------------------------

        partner_type = (
            "footman"
            if unit_type == "archer"
            else "archer"
        )

        partner = self._find_ally_type(
            observation,
            partner_type,
        )

        if self.state == State.ASSEMBLE:

            # Do not abandon assembly merely because an enemy
            # becomes visible.
            if partner is None:
                return

            distance = tile.distance(
                me.position(observation),
                partner["position"],
            )

            if (
                distance <= 3
                and self._at_rally_area(observation)
            ):
                if enemies:
                    self.state = State.ENGAGE
                else:
                    self.state = State.ADVANCE

            return

        if self.state == State.ADVANCE:

            if partner is None:
                self.state = State.ASSEMBLE

            elif enemies:
                self.state = State.ENGAGE

            return

        if self.state == State.ENGAGE:

            if partner is None:
                self.state = State.ASSEMBLE

            elif not enemies:
                self.state = State.ADVANCE

            return

    # ============================================================
    # ARCHER
    # ============================================================

    def _act_archer(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        enemies = visible.enemies(observation)
        here = me.position(observation)

        footman = self._find_ally_type(
            observation,
            "footman",
        )

        # --------------------------------------------------------
        # EMERGENCY: ENEMY CAVALRY TOO CLOSE
        # --------------------------------------------------------

        danger = self._dangerous_cavalry(
            observation,
        )

        if danger is not None and footman is not None:

            path = self._path_toward(
                observation,
                footman["position"],
                max_steps=2,
            )

            if path != 0:
                return action.move(
                    path,
                    danger["unit_id"],
                    observation,
                )

            return action.stay(
                danger["unit_id"],
                observation,
            )

        # --------------------------------------------------------
        # ASSEMBLE
        # --------------------------------------------------------

        if self.state == State.ASSEMBLE:

            if footman is not None:
                goal = footman["position"]
            else:
                goal = self._rally_point(
                    observation,
                )

            path = self._path_toward_archer_terrain(
                observation,
                goal,
                footman,
            )

            if path != 0:

                if enemies:
                    target = self._choose_target(
                        observation,
                        enemies,
                    )

                    return action.move(
                        path,
                        target["unit_id"],
                        observation,
                    )

                return action.move(path)

            if enemies:

                target = self._choose_target(
                    observation,
                    enemies,
                )

                return action.stay(
                    target["unit_id"],
                    observation,
                )

            return action.stay()

        # --------------------------------------------------------
        # ENGAGE
        # --------------------------------------------------------

        if self.state == State.ENGAGE and enemies:

            target = self._choose_target(
                observation,
                enemies,
            )

            # First priority: do not lose the footman entirely.
            if footman is not None:

                distance_to_footman = tile.distance(
                    here,
                    footman["position"],
                )

                if distance_to_footman > 4:

                    path = self._path_toward(
                        observation,
                        footman["position"],
                        max_steps=2,
                    )

                    if path != 0:
                        return action.move(
                            path,
                            target["unit_id"],
                            observation,
                        )

            # If we are protected, actively seek better terrain.
            if self._archer_is_protected(observation):

                path = self._best_protected_archer_position(
                    observation,
                    target["position"],
                )

                if path != 0:
                    return action.move(
                        path,
                        target["unit_id"],
                        observation,
                    )

            # Otherwise hold position and shoot.
            return action.stay(
                target["unit_id"],
                observation,
            )

        # --------------------------------------------------------
        # ADVANCE
        # --------------------------------------------------------

        if footman is not None:

            distance_to_footman = tile.distance(
                here,
                footman["position"],
            )

            if distance_to_footman > 4:
                goal = footman["position"]
            else:
                goal = tile.at_center(
                    observation,
                )

        else:
            goal = self._rally_point(
                observation,
            )

        path = self._path_toward_archer_terrain(
            observation,
            goal,
            footman,
        )

        if path != 0:
            return action.move(path)

        return action.stay()

    # ============================================================
    # FOOTMAN
    # ============================================================

    def _act_footman(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        enemies = visible.enemies(observation)

        archer = self._find_ally_type(
            observation,
            "archer",
        )

        # --------------------------------------------------------
        # ASSEMBLE
        # --------------------------------------------------------

        if self.state == State.ASSEMBLE:

            if archer is not None:
                goal = archer["position"]
            else:
                goal = self._rally_point(
                    observation,
                )

            path = self._path_toward(
                observation,
                goal,
                max_steps=2,
            )

            if path != 0:

                if enemies:

                    target = self._choose_target(
                        observation,
                        enemies,
                    )

                    return action.move(
                        path,
                        target["unit_id"],
                        observation,
                    )

                return action.move(path)

            if enemies:

                target = self._choose_target(
                    observation,
                    enemies,
                )

                return action.stay(
                    target["unit_id"],
                    observation,
                )

            return action.stay()

        # --------------------------------------------------------
        # ENGAGE
        # --------------------------------------------------------

        if self.state == State.ENGAGE and enemies:

            target = self._choose_target(
                observation,
                enemies,
            )

            path = self._path_toward(
                observation,
                target["position"],
                max_steps=2,
            )

            if path != 0:
                return action.move(
                    path,
                    target["unit_id"],
                    observation,
                )

            return action.stay(
                target["unit_id"],
                observation,
            )

        # --------------------------------------------------------
        # ADVANCE
        # --------------------------------------------------------

        if archer is None:
            goal = self._rally_point(
                observation,
            )
        else:
            goal = tile.at_center(
                observation,
            )

        path = self._path_toward(
            observation,
            goal,
            max_steps=2,
        )

        if path != 0:
            return action.move(path)

        return action.stay()

    # ============================================================
    # CAVALRY
    # ============================================================

    def _act_cavalry(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        enemies = visible.enemies(observation)

        # --------------------------------------------------------
        # ENGAGE
        # --------------------------------------------------------

        if self.state == State.ENGAGE and enemies:

            target = self._choose_target(
                observation,
                enemies,
            )

            path = self._path_toward(
                observation,
                target["position"],
            )

            if path != 0:
                return action.move(
                    path,
                    target["unit_id"],
                    observation,
                )

            return action.stay(
                target["unit_id"],
                observation,
            )

        # --------------------------------------------------------
        # REGROUP
        # --------------------------------------------------------

        if self.state == State.REGROUP:

            allies = visible.allies(
                observation
            )

            if allies:

                here = me.position(observation)

                nearest_ally = min(
                    allies,
                    key=lambda ally: tile.distance(
                        here,
                        ally["position"],
                    ),
                )

                goal = nearest_ally["position"]

            else:

                goal = self._rally_point(
                    observation,
                )

            path = self._path_toward(
                observation,
                goal,
            )

            if enemies:

                target = self._choose_target(
                    observation,
                    enemies,
                )

                if path != 0:
                    return action.move(
                        path,
                        target["unit_id"],
                        observation,
                    )

                return action.stay(
                    target["unit_id"],
                    observation,
                )

            if path != 0:
                return action.move(path)

            return action.stay()

        # --------------------------------------------------------
        # ASSEMBLE
        # --------------------------------------------------------

        if self.state == State.ASSEMBLE:

            rally = self._rally_point(
                observation,
            )

            path = self._path_toward(
                observation,
                rally,
            )

            if path != 0:
                return action.move(path)

            return action.stay()

        # --------------------------------------------------------
        # ADVANCE WITH GROUP
        # --------------------------------------------------------

        if self.state == State.ADVANCE:

            allies = visible.allies(
                observation
            )

            if allies:

                here = me.position(observation)

                nearest_ally = min(
                    allies,
                    key=lambda ally: tile.distance(
                        here,
                        ally["position"],
                    ),
                )

                distance_to_ally = tile.distance(
                    here,
                    nearest_ally["position"],
                )

                if distance_to_ally <= 3:
                    return action.stay()

                path = self._path_toward(
                    observation,
                    nearest_ally["position"],
                    max_steps=2,
                )

                if path != 0:
                    return action.move(path)

            return action.stay()

        # --------------------------------------------------------
        # SCOUT
        # --------------------------------------------------------

        center = tile.at_center(
            observation,
        )

        path = self._path_toward(
            observation,
            center,
        )

        if path != 0:
            return action.move(path)

        return action.stay()

    # ============================================================
    # TARGET SELECTION
    # ============================================================

    def _choose_target(
        self,
        observation: SkirmishObservation,
        enemies: list[dict],
    ) -> dict:

        here = me.position(observation)

        return min(
            enemies,
            key=lambda enemy: (
                enemy["hit_points"],
                tile.distance(
                    here,
                    enemy["position"],
                ),
                enemy["unit_id"],
            ),
        )

    # ============================================================
    # CAVALRY THREAT TO ARCHER
    # ============================================================

    def _dangerous_cavalry(
        self,
        observation: SkirmishObservation,
    ):

        here = me.position(observation)

        cavalry = [
            enemy
            for enemy in visible.enemies(observation)
            if enemy["type"] == "cavalry"
        ]

        if not cavalry:
            return None

        nearest = min(
            cavalry,
            key=lambda enemy: tile.distance(
                here,
                enemy["position"],
            ),
        )

        if tile.distance(
            here,
            nearest["position"],
        ) <= 3:
            return nearest

        return None

    # ============================================================
    # GROUP HELPERS
    # ============================================================

    def _find_ally_type(
        self,
        observation: SkirmishObservation,
        wanted_type: str,
    ):

        for ally in visible.allies(
            observation
        ):

            if ally["type"] == wanted_type:
                return ally

        return None

    def _has_nearby_ally(
        self,
        observation: SkirmishObservation,
        distance: int = 3,
    ) -> bool:

        here = me.position(observation)

        for ally in visible.allies(
            observation
        ):

            if tile.distance(
                here,
                ally["position"],
            ) <= distance:
                return True

        return False

    def _ally_can_support(
        self,
        ally: dict,
        target: dict,
    ) -> bool:

        stats = units.STATS[
            ally["type"]
        ]

        distance = tile.distance(
            ally["position"],
            target["position"],
        )

        reach = (
            stats.movement_points
            + stats.attack_range
        )

        return distance <= reach

    # ============================================================
    # ARCHER PROTECTION
    # ============================================================

    def _archer_is_protected(
        self,
        observation: SkirmishObservation,
    ) -> bool:
        """
        Archer is considered protected when another friendly
        unit is within 4 hexes.
        """

        here = me.position(observation)

        for ally in visible.allies(
            observation
        ):

            if tile.distance(
                here,
                ally["position"],
            ) <= 4:
                return True

        return False

    # ============================================================
    # TERRAIN SCORE
    # ============================================================

    def _terrain_score(
        self,
        observation: SkirmishObservation,
        position: AxialPosition,
    ) -> int:

        terrain_info = tile.terrain_at(
            observation,
            position,
        )

        terrain_type = terrain_info["terrain"]
        feature = terrain_info["feature"]

        if terrain_type in (
            "water",
            "void",
        ):
            return -100

        score = 0

        if terrain_type == "hill":
            score += 5

        if feature == "forest":
            score += 3

        if feature == "marsh":
            score -= 4

        return score

    # ============================================================
    # ARCHER FIRING POSITION SCORE
    # ============================================================

    def _archer_position_score(
        self,
        observation: SkirmishObservation,
        position: AxialPosition,
        target_position: AxialPosition,
        allies: list[dict],
    ) -> int:

        terrain_info = tile.terrain_at(
            observation,
            position,
        )

        terrain_type = terrain_info["terrain"]
        feature = terrain_info["feature"]

        if terrain_type in (
            "water",
            "void",
        ):
            return -1000

        score = 0

        # Hills are by far the best archer position.
        if terrain_type == "hill":
            score += 20

        if feature == "forest":
            score += 8

        if feature == "marsh":
            score -= 10

        # Prefer staying near friendly protection.
        if allies:

            nearest_ally_distance = min(
                tile.distance(
                    position,
                    ally["position"],
                )
                for ally in allies
            )

            if nearest_ally_distance <= 2:
                score += 10

            elif nearest_ally_distance <= 4:
                score += 5

            else:
                score -= 15

        # Prefer using the archer's range.
        target_distance = tile.distance(
            position,
            target_position,
        )

        if target_distance >= 4:
            score += 4

        elif target_distance <= 2:
            score -= 6

        return score

    # ============================================================
    # BEST ARCHER COMBAT POSITION
    # ============================================================

    def _best_protected_archer_position(
        self,
        observation: SkirmishObservation,
        target_position: AxialPosition,
    ) -> int:

        here = me.position(observation)

        attack_range = units.STATS[
            "archer"
        ].attack_range

        allies = visible.allies(
            observation
        )

        best_path = 0

        best_score = self._archer_position_score(
            observation,
            here,
            target_position,
            allies,
        )

        for path in action.legal_paths(
            observation
        ):

            if path == 0:
                continue

            directions = paths.decode(
                path
            )

            if len(directions) > 2:
                continue

            end_position = tile.at_path_end(
                here,
                path,
            )

            # Must still be able to fire at the target.
            if tile.distance(
                end_position,
                target_position,
            ) > attack_range:
                continue

            score = self._archer_position_score(
                observation,
                end_position,
                target_position,
                allies,
            )

            if score > best_score:
                best_score = score
                best_path = path

        return best_path

    # ============================================================
    # ARCHER TERRAIN-AWARE MOVEMENT
    # ============================================================

    def _path_toward_archer_terrain(
        self,
        observation: SkirmishObservation,
        goal: AxialPosition,
        footman,
    ) -> int:

        here = me.position(
            observation
        )

        best_path = 0
        best_score = -10000

        current_distance = tile.distance(
            here,
            goal,
        )

        for path in action.legal_paths(
            observation
        ):

            if path == 0:
                continue

            directions = paths.decode(
                path
            )

            if len(directions) > 2:
                continue

            end_position = tile.at_path_end(
                here,
                path,
            )

            goal_distance = tile.distance(
                end_position,
                goal,
            )

            # Do not move backwards just to chase terrain.
            if goal_distance > current_distance:
                continue

            terrain_info = tile.terrain_at(
                observation,
                end_position,
            )

            terrain_type = terrain_info["terrain"]
            feature = terrain_info["feature"]

            if terrain_type in (
                "water",
                "void",
            ):
                continue

            score = 0

            # Progress toward objective.
            score += (
                current_distance
                - goal_distance
            ) * 10

            # Strong terrain preference.
            if terrain_type == "hill":
                score += 15

            if feature == "forest":
                score += 6

            if feature == "marsh":
                score -= 8

            # Don't abandon the footman.
            if footman is not None:

                distance_to_footman = tile.distance(
                    end_position,
                    footman["position"],
                )

                if distance_to_footman <= 3:
                    score += 5

                elif distance_to_footman > 5:
                    score -= 20

            if score > best_score:
                best_score = score
                best_path = path

        return best_path

    # ============================================================
    # RALLY POINT
    # ============================================================

    def _rally_point(
        self,
        observation: SkirmishObservation,
    ) -> AxialPosition:

        center = tile.at_center(
            observation,
        )

        opposite = {
            1: 4,
            2: 5,
            3: 6,
            4: 1,
            5: 2,
            6: 3,
        }

        own_direction = opposite[
            me.direction(observation)
        ]

        anchor = center

        # Four tiles inside our side.
        for _ in range(4):

            anchor = tile.neighbors(
                anchor
            )[own_direction]

        candidates = self._positions_within(
            anchor,
            radius=2,
        )

        best_position = anchor
        best_score = -1000

        for position in candidates:

            terrain_score = self._terrain_score(
                observation,
                position,
            )

            if terrain_score <= -100:
                continue

            score = (
                terrain_score * 10
                - tile.distance(
                    position,
                    anchor,
                )
            )

            if score > best_score:
                best_score = score
                best_position = position

        return best_position

    def _at_rally_area(
        self,
        observation: SkirmishObservation,
        radius: int = 2,
    ) -> bool:

        rally = self._rally_point(
            observation,
        )

        return tile.distance(
            me.position(observation),
            rally,
        ) <= radius

    def _positions_within(
        self,
        center: AxialPosition,
        radius: int,
    ) -> list[AxialPosition]:

        results = [center]
        frontier = [center]

        seen = {
            (
                center["q"],
                center["r"],
            )
        }

        for _ in range(radius):

            new_frontier = []

            for position in frontier:

                for neighbor in tile.neighbors(
                    position
                ).values():

                    key = (
                        neighbor["q"],
                        neighbor["r"],
                    )

                    if key in seen:
                        continue

                    seen.add(key)
                    results.append(
                        neighbor
                    )
                    new_frontier.append(
                        neighbor
                    )

            frontier = new_frontier

        return results

    # ============================================================
    # GENERAL MOVEMENT
    # ============================================================

    def _path_toward(
        self,
        observation: SkirmishObservation,
        goal: AxialPosition,
        max_steps: int | None = None,
    ) -> int:

        here = me.position(
            observation
        )

        best_path = 0

        best_distance = tile.distance(
            here,
            goal,
        )

        best_terrain = self._terrain_score(
            observation,
            here,
        )

        for path in action.legal_paths(
            observation
        ):

            if path == 0:
                continue

            directions = paths.decode(
                path
            )

            if (
                max_steps is not None
                and len(directions) > max_steps
            ):
                continue

            end_position = tile.at_path_end(
                here,
                path,
            )

            distance = tile.distance(
                end_position,
                goal,
            )

            terrain_score = self._terrain_score(
                observation,
                end_position,
            )

            if distance < best_distance:

                best_path = path
                best_distance = distance
                best_terrain = terrain_score

            elif (
                distance == best_distance
                and terrain_score > best_terrain
            ):

                best_path = path
                best_terrain = terrain_score

        return best_path