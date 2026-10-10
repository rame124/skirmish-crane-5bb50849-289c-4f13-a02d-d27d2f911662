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
    ENGAGE = auto()
    REGROUP = auto()


class Agent:
    """
    Season 3 squad-based Crane Reach agent.

    Army organization:
      Squad 0: footman_0, archer_0, cavalry_0
      Squad 1: footman_1, archer_1, cavalry_1
      ...
      Squad 5: footman_5, archer_5, cavalry_5

      Reserve: footman_6 + footman_7

    Strategy:
    - Each squad gets its own rally lane.
    - Squads try to assemble before advancing.
    - Squad members slow down to stay together.
    - If enemies appear while allies are nearby, everyone attacks.
    - If a unit encounters enemies alone, it regroups.
    - Archers seek hills when protected by squadmates.
    - Cavalry prefers charge attacks when fighting with its squad.
    - Half-health cavalry regroups, but may rejoin combat once
      friendly support is present.
    - Movement can detour around rivers and other obstacles instead
      of freezing when no legal path directly reduces distance.
    """

    def reset(self, seed, observation) -> None:
        self.state = State.ASSEMBLE

        # Used to discourage immediate backtracking while navigating
        # around rivers and other obstacles.
        self.previous_position = None
        self.avoid_position = None

    # ============================================================
    # MAIN DECISION
    # ============================================================

    def act(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        current_position = me.position(observation)

        # Remember where this unit was on its previous activation.
        self.avoid_position = self.previous_position
        self.previous_position = {
            "q": current_position["q"],
            "r": current_position["r"],
        }

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

        enemies = visible.enemies(observation)
        unit_type = me.unit_type(observation)

        nearby_mates = self._nearby_squadmates(
            observation,
            distance=4,
        )

        # --------------------------------------------------------
        # CAVALRY DAMAGE RESPONSE
        # --------------------------------------------------------

        if unit_type == "cavalry":

            hp = me.hit_points(observation)
            max_hp = units.STATS["cavalry"].hit_points

            if hp <= max_hp / 2:

                # A wounded cavalry unit does not abandon a fight
                # when the squad is already with it.
                if enemies and nearby_mates:
                    self.state = State.ENGAGE
                else:
                    self.state = State.REGROUP

                return

        # --------------------------------------------------------
        # ENEMY CONTACT
        # --------------------------------------------------------

        if enemies:

            # We have squad support: fight together.
            if nearby_mates:
                self.state = State.ENGAGE

            # We encountered the enemy alone.
            else:
                self.state = State.REGROUP

            return

        # --------------------------------------------------------
        # ASSEMBLY
        # --------------------------------------------------------

        if self.state == State.ASSEMBLE:

            if self._squad_ready(observation):
                self.state = State.ADVANCE

            return

        # --------------------------------------------------------
        # AFTER COMBAT
        # --------------------------------------------------------

        if self.state == State.ENGAGE:

            self.state = State.ADVANCE
            return

        # --------------------------------------------------------
        # REGROUPING
        # --------------------------------------------------------

        if self.state == State.REGROUP:

            # Once we have at least one squadmate nearby,
            # resume moving with the group.
            if nearby_mates:
                self.state = State.ADVANCE

            return

        # --------------------------------------------------------
        # ADVANCING
        # --------------------------------------------------------

        if self.state == State.ADVANCE:

            # If we completely lose our squad, regroup.
            if not self._visible_squadmates(observation):
                self.state = State.REGROUP

    # ============================================================
    # ARCHER
    # ============================================================

    def _act_archer(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        enemies = visible.enemies(observation)

        # --------------------------------------------------------
        # ASSEMBLE / REGROUP
        # --------------------------------------------------------

        if self.state in (
            State.ASSEMBLE,
            State.REGROUP,
        ):

            goal = self._formation_goal(
                observation,
            )

            path = self._path_toward_archer_terrain(
                observation,
                goal,
            )

            if path != 0:

                if enemies:
                    target = self._choose_target(
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
                enemies,
            )

            # If protected, try to obtain high ground before
            # simply standing still and firing.
            if self._archer_is_protected(
                observation
            ):

                path = self._best_archer_attack_position(
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
        # ADVANCE
        # --------------------------------------------------------

        goal = self._advance_goal(
            observation,
        )

        path = self._path_toward_archer_terrain(
            observation,
            goal,
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

        # --------------------------------------------------------
        # ASSEMBLE / REGROUP
        # --------------------------------------------------------

        if self.state in (
            State.ASSEMBLE,
            State.REGROUP,
        ):

            goal = self._formation_goal(
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

        goal = self._advance_goal(
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
        # ASSEMBLE / REGROUP
        # --------------------------------------------------------

        if self.state in (
            State.ASSEMBLE,
            State.REGROUP,
        ):

            goal = self._formation_goal(
                observation,
            )

            # Full speed is okay while trying to catch the squad.
            path = self._path_toward(
                observation,
                goal,
            )

            if path != 0:

                if enemies:

                    target = self._choose_target(
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
                    enemies,
                )

                return action.stay(
                    target["unit_id"],
                    observation,
                )

            return action.stay()

        # --------------------------------------------------------
        # ENGAGE
        #
        # Cavalry attempts to get a Charge attack when possible.
        # --------------------------------------------------------

        if self.state == State.ENGAGE and enemies:

            target = self._choose_target(
                enemies,
            )

            path = self._cavalry_attack_path(
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
        # ADVANCE
        #
        # Once cavalry is with the squad, limit it to two steps.
        # This prevents it from outrunning the footman/archer.
        # --------------------------------------------------------

        goal = self._advance_goal(
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
    # SQUAD IDENTITY
    # ============================================================

    def _unit_index(
        self,
        observation: SkirmishObservation,
    ) -> int:

        unit_id = me.unit_id(observation)

        try:
            return int(
                unit_id.rsplit("_", 1)[1]
            )
        except (ValueError, IndexError):
            return 0

    def _squad_number(
        self,
        observation: SkirmishObservation,
    ) -> int:
        """
        Squads 0-5 are F/A/C teams.

        Footmen 6 and 7 form reserve squad 6.
        """

        index = self._unit_index(
            observation
        )

        unit_type = me.unit_type(
            observation
        )

        if (
            unit_type == "footman"
            and index >= 6
        ):
            return 6

        return min(index, 5)

    def _squad_member_ids(
        self,
        observation: SkirmishObservation,
    ) -> set[str]:

        side = me.side(observation)

        squad = self._squad_number(
            observation
        )

        # Reserve footmen.
        if squad == 6:

            return {
                f"{side}_footman_6",
                f"{side}_footman_7",
            }

        # Normal 3-unit squad.
        return {
            f"{side}_footman_{squad}",
            f"{side}_archer_{squad}",
            f"{side}_cavalry_{squad}",
        }

    def _visible_squadmates(
        self,
        observation: SkirmishObservation,
    ) -> list[dict]:

        squad_ids = self._squad_member_ids(
            observation
        )

        own_id = me.unit_id(
            observation
        )

        return [
            ally
            for ally in visible.allies(observation)
            if (
                ally["unit_id"] in squad_ids
                and ally["unit_id"] != own_id
            )
        ]

    def _nearby_squadmates(
        self,
        observation: SkirmishObservation,
        distance: int = 4,
    ) -> list[dict]:

        here = me.position(observation)

        return [
            ally
            for ally in self._visible_squadmates(
                observation
            )
            if tile.distance(
                here,
                ally["position"],
            ) <= distance
        ]

    def _squad_ready(
        self,
        observation: SkirmishObservation,
    ) -> bool:

        squad = self._squad_number(
            observation
        )

        nearby = self._nearby_squadmates(
            observation,
            distance=4,
        )

        # Reserve pair needs its one partner.
        if squad == 6:
            return len(nearby) >= 1

        # Normal squad needs both other members.
        if len(nearby) >= 2:
            return True

        # Don't wait forever if terrain/pathing separates one member.
        current_round = self._round_number(
            observation
        )

        if (
            current_round >= 10
            and len(nearby) >= 1
        ):
            return True

        return False

    # ============================================================
    # FORMATION / MOVEMENT GOALS
    # ============================================================

    def _formation_goal(
        self,
        observation: SkirmishObservation,
    ) -> AxialPosition:
        """
        During assembly/regrouping:
        - Follow a visible squadmate if separated.
        - Otherwise return to the squad's rally area.
        """

        mates = self._visible_squadmates(
            observation
        )

        if mates:

            here = me.position(
                observation
            )

            nearest = min(
                mates,
                key=lambda ally: tile.distance(
                    here,
                    ally["position"],
                ),
            )

            return nearest["position"]

        return self._squad_rally_point(
            observation
        )

    def _advance_goal(
        self,
        observation: SkirmishObservation,
    ) -> AxialPosition:
        """
        If a squadmate is falling behind, maintain cohesion.
        Otherwise advance down the squad's lane.
        """

        here = me.position(
            observation
        )

        mates = self._visible_squadmates(
            observation
        )

        if mates:

            nearest = min(
                mates,
                key=lambda ally: tile.distance(
                    here,
                    ally["position"],
                ),
            )

            nearest_distance = tile.distance(
                here,
                nearest["position"],
            )

            if nearest_distance > 4:
                return nearest["position"]

        rally = self._squad_rally_point(
            observation
        )

        # Mirror the rally point into enemy territory.
        # This naturally creates a lane for each squad.
        return tile.at_mirror(
            rally,
            observation,
        )

    # ============================================================
    # SQUAD RALLY POINT
    # ============================================================

    def _squad_rally_point(
        self,
        observation: SkirmishObservation,
    ) -> AxialPosition:

        center = tile.at_center(
            observation
        )

        forward = me.direction(
            observation
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
            forward
        ]

        # Adjust automatically to the season's map size.
        extent = self._field_extent(
            observation
        )

        rally_depth = max(
            4,
            extent // 2,
        )

        anchor = center

        for _ in range(rally_depth):

            anchor = tile.neighbors(
                anchor
            )[own_direction]

        squad = self._squad_number(
            observation
        )

        # Six squads spread across six lanes.
        lane_offsets = {
            0: -3,
            1: -2,
            2: -1,
            3: 1,
            4: 2,
            5: 3,
            6: 0,
        }

        offset = lane_offsets.get(
            squad,
            0,
        )

        # Mirror lateral orientation between Red and Blue.
        if forward == 2:
            negative_direction = 6
            positive_direction = 3
        else:
            negative_direction = 3
            positive_direction = 6

        lateral_direction = (
            positive_direction
            if offset > 0
            else negative_direction
        )

        for _ in range(abs(offset)):

            anchor = tile.neighbors(
                anchor
            )[lateral_direction]

        # Each squad looks only around its own local anchor.
        # That prevents the entire army from chasing one hill.
        candidates = self._positions_within(
            anchor,
            radius=1,
        )

        best_position = anchor
        best_score = -10000

        for position in candidates:

            score = self._terrain_score(
                observation,
                position,
            )

            if score <= -100:
                continue

            weighted_score = (
                score * 10
                - tile.distance(
                    anchor,
                    position,
                )
            )

            if weighted_score > best_score:

                best_score = weighted_score
                best_position = position

        return best_position

    # ============================================================
    # TARGET SELECTION
    # ============================================================

    def _choose_target(
        self,
        enemies: list[dict],
    ) -> dict:
        """
        Shared deterministic rule.

        Lowest HP first, then unit ID.

        Removing distance as a tie-breaker makes squad members
        more likely to select the exact same enemy.
        """

        return min(
            enemies,
            key=lambda enemy: (
                enemy["hit_points"],
                enemy["unit_id"],
            ),
        )

    # ============================================================
    # ARCHER SUPPORT / HIGH GROUND
    # ============================================================

    def _archer_is_protected(
        self,
        observation: SkirmishObservation,
    ) -> bool:

        return bool(
            self._nearby_squadmates(
                observation,
                distance=4,
            )
        )

    def _best_archer_attack_position(
        self,
        observation: SkirmishObservation,
        target_position: AxialPosition,
    ) -> int:
        """
        Seek high ground while:
        - remaining in attack range,
        - remaining near a squadmate,
        - avoiding marsh when possible.
        """

        here = me.position(
            observation
        )

        allies = self._visible_squadmates(
            observation
        )

        attack_range = units.STATS[
            "archer"
        ].attack_range

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

            end = tile.at_path_end(
                here,
                path,
            )

            # Must keep the target in shooting range.
            if tile.distance(
                end,
                target_position,
            ) > attack_range:
                continue

            # Must remain close enough to at least one squadmate.
            if allies:

                ally_distance = min(
                    tile.distance(
                        end,
                        ally["position"],
                    )
                    for ally in allies
                )

                if ally_distance > 4:
                    continue

            score = self._archer_position_score(
                observation,
                end,
                target_position,
                allies,
            )

            if score > best_score:

                best_score = score
                best_path = path

        return best_path

    def _archer_position_score(
        self,
        observation: SkirmishObservation,
        position: AxialPosition,
        target_position: AxialPosition,
        allies: list[dict],
    ) -> int:

        info = tile.terrain_at(
            observation,
            position,
        )

        terrain_type = info["terrain"]
        feature = info["feature"]

        if terrain_type in (
            "water",
            "void",
        ):
            return -10000

        score = 0

        # High ground is very valuable to the archer.
        if terrain_type == "hill":
            score += 30

        if feature == "forest":
            score += 8

        if feature == "marsh":
            score -= 10

        # Stay protected.
        if allies:

            nearest_ally = min(
                tile.distance(
                    position,
                    ally["position"],
                )
                for ally in allies
            )

            if nearest_ally <= 2:
                score += 10

            elif nearest_ally <= 4:
                score += 5

            else:
                score -= 20

        # Use range rather than standing beside melee enemies.
        target_distance = tile.distance(
            position,
            target_position,
        )

        if target_distance >= 4:
            score += 5

        elif target_distance <= 2:
            score -= 8

        return score

    def _path_toward_archer_terrain(
        self,
        observation: SkirmishObservation,
        goal: AxialPosition,
    ) -> int:

        here = me.position(
            observation
        )

        current_distance = tile.distance(
            here,
            goal,
        )

        mates = self._visible_squadmates(
            observation
        )

        best_path = 0
        best_score = -10000

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

            end = tile.at_path_end(
                here,
                path,
            )

            goal_distance = tile.distance(
                end,
                goal,
            )

            # Positive = progress.
            # Negative is allowed so the archer can detour around rivers.
            progress = current_distance - goal_distance

            terrain = tile.terrain_at(
                observation,
                end,
            )

            if terrain["terrain"] in (
                "water",
                "void",
            ):
                continue

            score = progress * 12

            # Discourage immediately reversing last activation.
            if self.avoid_position is not None:
                if (
                    end["q"] == self.avoid_position["q"]
                    and end["r"] == self.avoid_position["r"]
                ):
                    score -= 30

            if terrain["terrain"] == "hill":
                score += 15

            if terrain["feature"] == "forest":
                score += 6

            if terrain["feature"] == "marsh":
                score -= 8

            if mates:

                nearest_mate = min(
                    tile.distance(
                        end,
                        ally["position"],
                    )
                    for ally in mates
                )

                if nearest_mate <= 3:
                    score += 8

                elif nearest_mate > 5:
                    score -= 20

            if score > best_score:

                best_score = score
                best_path = path

        return best_path

    # ============================================================
    # CAVALRY CHARGE
    # ============================================================

    def _cavalry_attack_path(
        self,
        observation: SkirmishObservation,
        target_position: AxialPosition,
    ) -> int:
        """
        Prefer getting adjacent to the target.

        When two paths are similarly good, prefer one whose final
        tile is at least 3 hexes from the starting tile, activating
        Season 3's cavalry Charge ability.
        """

        here = me.position(
            observation
        )

        best_path = 0
        best_score = -10000

        start_distance = tile.distance(
            here,
            target_position,
        )

        for path in action.legal_paths(
            observation
        ):

            if path == 0:
                continue

            end = tile.at_path_end(
                here,
                path,
            )

            target_distance = tile.distance(
                end,
                target_position,
            )

            displacement = tile.distance(
                here,
                end,
            )

            # Getting close is the main goal.
            score = -target_distance * 100

            # Strongly reward a legal charge.
            if displacement >= 3:
                score += 35

            # Prefer useful terrain if otherwise similar.
            score += self._terrain_score(
                observation,
                end,
            )

            # Don't choose movement that actually moves farther away
            # unless no closer choice exists.
            if target_distance > start_distance:
                score -= 100

            if score > best_score:

                best_score = score
                best_path = path

        return best_path

    # ============================================================
    # TERRAIN
    # ============================================================

    def _terrain_score(
        self,
        observation: SkirmishObservation,
        position: AxialPosition,
    ) -> int:

        info = tile.terrain_at(
            observation,
            position,
        )

        terrain_type = info["terrain"]
        feature = info["feature"]

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
    # GENERAL MOVEMENT
    # ============================================================

    def _path_toward(
        self,
        observation: SkirmishObservation,
        goal: AxialPosition,
        max_steps: int | None = None,
    ) -> int:
        """
        Move toward the goal when possible.

        If no legal path gets closer, allow a sideways/detour move.
        This lets units navigate around rivers instead of freezing
        whenever the direct route is blocked.

        Recent positions are penalized to reduce back-and-forth
        movement along obstacles.
        """

        here = me.position(observation)

        current_distance = tile.distance(
            here,
            goal,
        )

        improving_paths = []
        detour_paths = []

        for path in action.legal_paths(
            observation
        ):

            if path == 0:
                continue

            directions = paths.decode(path)

            if (
                max_steps is not None
                and len(directions) > max_steps
            ):
                continue

            end = tile.at_path_end(
                here,
                path,
            )

            distance = tile.distance(
                end,
                goal,
            )

            terrain_score = self._terrain_score(
                observation,
                end,
            )

            # Strongly discourage immediately undoing our
            # previous movement.
            backtrack_penalty = 0

            if self.avoid_position is not None:
                if (
                    end["q"] == self.avoid_position["q"]
                    and end["r"] == self.avoid_position["r"]
                ):
                    backtrack_penalty = 50

            path_length = len(directions)

            score = (
                -distance * 100
                + terrain_score * 3
                - path_length
                - backtrack_penalty
            )

            if distance < current_distance:
                improving_paths.append(
                    (score, path)
                )
            else:
                detour_paths.append(
                    (score, path)
                )

        # Normal progress toward goal.
        if improving_paths:

            improving_paths.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            return improving_paths[0][1]

        # Don't wander unnecessarily when already close.
        if current_distance <= 2:
            return 0

        # If nothing gets closer, take the least-bad detour.
        if detour_paths:

            detour_paths.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            return detour_paths[0][1]

        return 0

    # ============================================================
    # SMALL UTILITIES
    # ============================================================

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

            next_frontier = []

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

                    next_frontier.append(
                        neighbor
                    )

            frontier = next_frontier

        return results

    def _field_extent(
        self,
        observation: SkirmishObservation,
    ) -> int:

        try:
            return int(
                observation[
                    "observation"
                ][
                    "parameters"
                ][
                    "field_extent"
                ]
            )
        except (KeyError, TypeError, ValueError):
            return 12

    def _round_number(
        self,
        observation: SkirmishObservation,
    ) -> int:

        try:
            return int(
                observation[
                    "observation"
                ][
                    "round"
                ]
            )
        except (KeyError, TypeError, ValueError):
            return 1
