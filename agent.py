from enum import Enum, auto

from sandbox.crane import action, me, paths, roster, tile, units, visible, zone
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
    CAPTURE = auto()


class Agent:
    """
    Season 4 dynamic local-squad Crane Reach agent.

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
    - Archers prioritize firing uptime, then tactically reposition for high-ground damage.
    - Cavalry prefers reachable attacks first, then supported charge opportunities; distant archers cannot override hittable nearby enemies.
    - Half-health cavalry regroups, but may rejoin combat once
      friendly support is present.
    - Movement can detour around rivers and other obstacles instead
      of freezing when no legal path directly reduces distance.
    - Units dynamically cooperate with nearby allies and communicate enemy targets/last-known positions.
    - Footmen elect exactly one home defender; a footman already on the home zone wins automatically, while all others reinforce the middle objective.
    """

    def reset(self, seed, observation) -> None:
        self.state = State.ASSEMBLE

        # Used to discourage immediate backtracking while navigating
        # around rivers and other obstacles.
        self.previous_position = None
        self.avoid_position = None

        # Keep several recent hexes, not just the previous one. This is
        # used to break two-hex movement loops around allied blockers.
        self.recent_positions = []

        # Count consecutive activations where this unit has not changed
        # hexes. This lets cavalry and crowded melee units deliberately
        # break out of local traffic jams instead of passing forever.
        self.stationary_turns = 0

        # ------------------------------------------------------------
        # Season 4 communication state
        # ------------------------------------------------------------
        # Messages arrive after act(), so information received in chat()
        # is saved here for the NEXT activation.
        self.current_round = 1
        self.shared_target_id = None
        self.shared_enemy_position = None
        self.shared_enemy_round = -999
        self.pending_messages = []
        self.last_announced_target = None
        self.last_announced_round = -999

        # Dynamic squads are based on proximity, not fixed unit IDs.
        # Cache only unit_id -> player so we can directly message the
        # allies that are actually closest on the current activation.
        self.ally_player_by_unit_id = {}

        for entry in roster.allies(observation):
            unit_id = entry.get("unit_id")
            player = entry.get("player")

            if unit_id is not None and player is not None:
                self.ally_player_by_unit_id[unit_id] = player

        # Season 5: exactly one footman is responsible for holding the
        # capture zone on our side of the battlefield. At spawn, allied
        # footmen are close enough to compare starting distances.
        self.home_defender_id = self._select_home_defender(
            observation
        )

        # Distributed home-guard election.
        #
        # Each footman announces:
        #   - whether it STARTED on our home capture zone,
        #   - its starting distance to that zone.
        #
        # "Already on the zone" always wins, then shortest distance,
        # then unit ID for a deterministic tie-break.
        self.home_guard_priority = None

        if me.unit_type(observation) == "footman":

            area = self._home_capture_zone(
                observation
            )

            if area is not None:

                here = me.position(
                    observation
                )

                distance = tile.distance(
                    here,
                    area["center"],
                )

                on_home_zone = (
                    zone.at(
                        observation,
                        here,
                    ) is not None
                    and distance <= 2
                )

                self.home_guard_priority = (
                    0 if on_home_zone else 1,
                    distance,
                    me.unit_id(observation),
                )

                claim = (
                    f"G|"
                    f"{me.unit_id(observation)}|"
                    f"{1 if on_home_zone else 0}|"
                    f"{distance}"
                )

                # Send the claim to every allied footman for which we
                # have a player mapping. Unit IDs expose unit type.
                for unit_id, player in (
                    self.ally_player_by_unit_id.items()
                ):

                    if "_footman_" not in unit_id:
                        continue

                    self.pending_messages.append(
                        {
                            "to": player,
                            "text": claim,
                        }
                    )

    # ============================================================
    # MAIN DECISION
    # ============================================================

    def act(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        current_position = me.position(observation)
        self.current_round = self._round_number(observation)

        # Count repeated activations on the same hex before updating the
        # remembered position.
        if self.previous_position is not None:
            if (
                current_position["q"] == self.previous_position["q"]
                and current_position["r"] == self.previous_position["r"]
            ):
                self.stationary_turns += 1
            else:
                self.stationary_turns = 0

        # Remember where this unit was on its previous activation.
        self.avoid_position = self.previous_position
        self.previous_position = {
            "q": current_position["q"],
            "r": current_position["r"],
        }

        self.recent_positions.append(
            (
                current_position["q"],
                current_position["r"],
            )
        )

        # Four activations is enough to detect common A<->B and
        # A->B->C->A loops without preventing a legitimate later return.
        if len(self.recent_positions) > 4:
            self.recent_positions.pop(0)

        self._update_state(observation)

        # If this unit currently sees an enemy, tell its squad which
        # target to focus and where contact occurred. The message will
        # become useful to recipients on their next activation.
        enemies = visible.enemies(observation)
        if enemies:
            target = self._choose_target(enemies)
            self._remember_enemy(target, self.current_round)
            self._queue_enemy_report(observation, target)

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
        # CAVALRY
        #
        # Season 4 was too aggressive: healthy cavalry committed to
        # every visible enemy, even when completely alone.
        #
        # New rule:
        # - if an enemy is already in attack range, FIGHT;
        # - otherwise only commit when at least one nearby ally is
        #   available to support the attack;
        # - isolated cavalry regroups instead of solo-charging.
        # --------------------------------------------------------

        if unit_type == "cavalry":

            hp = me.hit_points(observation)
            max_hp = units.STATS["cavalry"].hit_points

            if enemies:

                here = me.position(observation)
                attack_range = units.STATS[
                    "cavalry"
                ].attack_range

                enemy_in_range = any(
                    tile.distance(
                        here,
                        enemy["position"],
                    ) <= attack_range
                    for enemy in enemies
                )

                if enemy_in_range:
                    self.state = State.ENGAGE

                elif nearby_mates:
                    self.state = State.ENGAGE

                else:
                    self.state = State.REGROUP

                return

            if hp <= max_hp / 2:
                self.state = State.REGROUP
                return

            # No contact: retain the Season 4 forward pressure.
            if self.state in (
                State.ASSEMBLE,
                State.REGROUP,
                State.ENGAGE,
            ):
                self.state = State.ADVANCE

            return

        # --------------------------------------------------------
        # FOOTMAN / ARCHER ENEMY CONTACT
        # --------------------------------------------------------

        if enemies:

            if nearby_mates:
                self.state = State.ENGAGE

            elif unit_type == "footman":
                self.state = State.ENGAGE

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

            if nearby_mates or self._shared_enemy_is_fresh():
                self.state = State.ADVANCE
                return

            if self._round_number(observation) >= 8:
                self.state = State.ADVANCE

            return

        return

    def _act_archer(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        enemies = visible.enemies(observation)
        here = me.position(observation)
        attack_range = units.STATS["archer"].attack_range

        # --------------------------------------------------------
        # VISIBLE ENEMY
        #
        # Priority order:
        # 1. Get into firing range.
        # 2. If already in range, improve firing terrain.
        # 3. Fire.
        # --------------------------------------------------------

        if enemies:

            target = self._choose_target(
                enemies,
            )

            target_distance = tile.distance(
                here,
                target["position"],
            )

            # A hill with no shot is worthless. Leave it immediately
            # and close toward the target until we can actually attack.
            if target_distance > attack_range:

                path = self._path_toward_archer_terrain(
                    observation,
                    target["position"],
                )

                if path == 0:
                    path = self._path_toward(
                        observation,
                        target["position"],
                        max_steps=2,
                        allow_close_detour=True,
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

            # We already have a shot. Now terrain matters.
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
        # NO VISIBLE ENEMY
        #
        # Never camp solely because current terrain is good.
        # Move toward fresh combat intel or continue the advance.
        # --------------------------------------------------------

        if self._shared_enemy_is_fresh():

            goal = {
                "q": self.shared_enemy_position["q"],
                "r": self.shared_enemy_position["r"],
            }

        else:

            goal = self._advance_goal(
                observation,
            )

        path = self._path_toward_archer_terrain(
            observation,
            goal,
        )

        if path == 0:
            path = self._path_toward(
                observation,
                goal,
                max_steps=2,
                allow_close_detour=True,
            )

        if path != 0:
            return action.move(path)

        return action.stay()

    def _act_footman(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        # Exactly one footman remains responsible for our home point.
        if (
            me.unit_id(observation)
            == self.home_defender_id
        ):
            return self._act_home_defender(
                observation
            )

        enemies = visible.enemies(observation)

        # --------------------------------------------------------
        # ACTIVE COMBAT
        # --------------------------------------------------------

        if enemies:

            target = self._choose_target(
                enemies,
            )

            path = self._path_toward(
                observation,
                target["position"],
                max_steps=2,
                allow_close_detour=True,
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
        # NO ENEMY
        #
        # Season 5 has three capture zones. Every non-defender footman
        # should contribute to the contested middle objective rather
        # than following a nearby ally back toward our home defender.
        # --------------------------------------------------------

        middle = self._middle_capture_zone(
            observation
        )

        if middle is not None:
            goal = middle["center"]
        else:
            goal = self._advance_goal(
                observation
            )

        path = self._path_toward(
            observation,
            goal,
            max_steps=2,
            allow_close_detour=True,
        )

        if path != 0:
            return action.move(path)

        return action.stay()

    def _act_cavalry(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        enemies = visible.enemies(observation)

        hp = me.hit_points(observation)
        max_hp = units.STATS["cavalry"].hit_points

        # --------------------------------------------------------
        # CONTACT
        # --------------------------------------------------------

        if enemies:

            here = me.position(observation)
            attack_range = units.STATS[
                "cavalry"
            ].attack_range

            # CRITICAL FIX:
            # If cavalry can already attack an enemy, attack NOW.
            # The old charge planner skipped path 0, so cavalry could
            # ride past an adjacent enemy while searching for a charge.
            attackable = [
                enemy
                for enemy in enemies
                if tile.distance(
                    here,
                    enemy["position"],
                ) <= attack_range
            ]

            if attackable:

                target = self._choose_cavalry_target(
                    observation,
                    attackable,
                )

                return action.stay(
                    target["unit_id"],
                    observation,
                )

            # ----------------------------------------------------
            # Supported engagement:
            # charge/close only when the state machine found support.
            # ----------------------------------------------------

            if self.state == State.ENGAGE:

                target = self._choose_cavalry_target(
                    observation,
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

                path = self._path_toward(
                    observation,
                    target["position"],
                    allow_close_detour=True,
                )

                if path != 0:
                    return action.move(
                        path,
                        target["unit_id"],
                        observation,
                    )

                return action.stay()

            # ----------------------------------------------------
            # Isolated contact:
            # do NOT solo-charge. Regroup with the nearest visible ally.
            # ----------------------------------------------------

            goal = self._formation_goal(
                observation,
            )

            path = self._path_toward(
                observation,
                goal,
                allow_close_detour=True,
            )

            if path != 0:
                return action.move(path)

            return action.stay()

        # --------------------------------------------------------
        # WOUNDED AND NO CONTACT
        # --------------------------------------------------------

        if hp <= max_hp / 2:

            goal = self._formation_goal(
                observation,
            )

            path = self._path_toward(
                observation,
                goal,
                allow_close_detour=True,
            )

            if path != 0:
                return action.move(path)

            return action.stay()

        # --------------------------------------------------------
        # HEALTHY, NO CONTACT
        #
        # Preserve the Season 4 forward movement. The restraint is only
        # applied when cavalry actually sees an enemy and lacks support.
        # --------------------------------------------------------

        if self._shared_enemy_is_fresh():

            goal = {
                "q": self.shared_enemy_position["q"],
                "r": self.shared_enemy_position["r"],
            }

        else:

            rally = self._squad_rally_point(
                observation,
            )

            goal = tile.at_mirror(
                rally,
                observation,
            )

        path = self._path_toward(
            observation,
            goal,
            allow_close_detour=True,
        )

        if path != 0:
            return action.move(path)

        return action.stay()

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

    def _visible_squadmates(
        self,
        observation: SkirmishObservation,
    ) -> list[dict]:
        """
        Form a temporary local squad every activation.

        This unit cooperates with the two closest allies it can
        currently see, regardless of unit type. This allows squads
        to reorganize naturally as terrain, combat, and casualties
        separate the army.
        """

        here = me.position(observation)
        allies = list(visible.allies(observation))

        allies.sort(
            key=lambda ally: (
                tile.distance(
                    here,
                    ally["position"],
                ),
                ally["unit_id"],
            )
        )

        return allies[:2]

    def _nearby_squadmates(
        self,
        observation: SkirmishObservation,
        distance: int = 4,
    ) -> list[dict]:

        here = me.position(observation)

        return [
            ally
            for ally in self._visible_squadmates(observation)
            if tile.distance(
                here,
                ally["position"],
            ) <= distance
        ]

    def _squad_ready(
        self,
        observation: SkirmishObservation,
    ) -> bool:
        """
        One nearby ally is enough to begin operating as a team.

        If a unit remains isolated for several rounds, allow it
        to advance so it does not become permanently stuck.
        """

        if self._nearby_squadmates(
            observation,
            distance=4,
        ):
            return True

        return self._round_number(observation) >= 8

    # ============================================================
    # FORMATION / MOVEMENT GOALS
    # ============================================================

    def _formation_goal(
        self,
        observation: SkirmishObservation,
    ) -> AxialPosition:
        """
        During assembly/regrouping, move toward the closest visible
        ally. If nobody is visible, head toward the friendly rally
        area until another unit comes into sight.
        """

        mates = self._visible_squadmates(observation)

        if mates:

            here = me.position(observation)

            nearest = min(
                mates,
                key=lambda ally: tile.distance(
                    here,
                    ally["position"],
                ),
            )

            return nearest["position"]

        return self._squad_rally_point(observation)

    def _advance_goal(
        self,
        observation: SkirmishObservation,
    ) -> AxialPosition:
        """
        Stay near the closest currently visible allies.

        Fresh communicated enemy intel takes priority when the
        local group has lost direct sight of the enemy.
        """

        here = me.position(observation)
        mates = self._visible_squadmates(observation)

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

        if self._shared_enemy_is_fresh():
            return {
                "q": self.shared_enemy_position["q"],
                "r": self.shared_enemy_position["r"],
            }

        rally = self._squad_rally_point(observation)

        return tile.at_mirror(
            rally,
            observation,
        )

    def _squad_rally_point(
        self,
        observation: SkirmishObservation,
    ) -> AxialPosition:
        """
        Shared fallback rally area on our side of the battlefield.

        Units only use this when they cannot see another ally.
        As soon as an ally becomes visible, dynamic grouping takes
        over and the unit follows its closest local partners.
        """

        center = tile.at_center(observation)
        forward = me.direction(observation)

        opposite = {
            1: 4,
            2: 5,
            3: 6,
            4: 1,
            5: 2,
            6: 3,
        }

        own_direction = opposite[forward]

        extent = self._field_extent(observation)
        rally_depth = max(
            4,
            extent // 2,
        )

        anchor = center

        for _ in range(rally_depth):
            anchor = tile.neighbors(
                anchor
            )[own_direction]

        # Look around the fallback area for useful terrain.
        candidates = self._positions_within(
            anchor,
            radius=2,
        )

        best_position = anchor
        best_score = -10000

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


    # ============================================================
    # SEASON 4 COMMUNICATION
    # ============================================================

    def chat(self, inbox: list[dict]) -> list[dict] | None:
        """
        Receive:
        - G|unit_id|on_home_zone|distance
          Home-guard election messages.
        - E|round|unit_id|q|r
          Delayed enemy reports.

        Messages affect the NEXT activation.
        """

        for message in inbox:

            text = str(
                message.get("text", "")
            )

            parts = text.split("|")

            if not parts:
                continue

            # ----------------------------------------------------
            # Home guard election
            # ----------------------------------------------------

            if (
                parts[0] == "G"
                and len(parts) == 4
            ):

                try:
                    unit_id = parts[1]
                    on_home = int(parts[2])
                    distance = int(parts[3])
                except (TypeError, ValueError):
                    continue

                priority = (
                    0 if on_home == 1 else 1,
                    distance,
                    unit_id,
                )

                if (
                    self.home_guard_priority is None
                    or priority
                    < self.home_guard_priority
                ):

                    self.home_guard_priority = (
                        priority
                    )

                    self.home_defender_id = (
                        unit_id
                    )

                continue

            # ----------------------------------------------------
            # Enemy report
            # ----------------------------------------------------

            if (
                parts[0] != "E"
                or len(parts) != 5
            ):
                continue

            try:
                report_round = int(parts[1])
                unit_id = parts[2]
                q = int(parts[3])
                r = int(parts[4])
            except (TypeError, ValueError):
                continue

            if report_round >= self.shared_enemy_round:

                self.shared_target_id = (
                    unit_id
                )

                self.shared_enemy_position = {
                    "q": q,
                    "r": r,
                }

                self.shared_enemy_round = (
                    report_round
                )

        outgoing = self.pending_messages
        self.pending_messages = []

        return outgoing if outgoing else None

    def _remember_enemy(
        self,
        enemy: dict,
        report_round: int,
    ) -> None:

        self.shared_target_id = enemy["unit_id"]
        self.shared_enemy_position = {
            "q": enemy["position"]["q"],
            "r": enemy["position"]["r"],
        }
        self.shared_enemy_round = report_round

    def _queue_enemy_report(
        self,
        observation: SkirmishObservation,
        enemy: dict,
    ) -> None:
        """
        Directly report contact to this unit's CURRENT local partners.

        Because the local squad is recalculated every activation,
        reports follow whichever allies are actually nearby instead
        of a fixed footman/archer/cavalry roster pairing.
        """

        target_id = enemy["unit_id"]

        if (
            target_id == self.last_announced_target
            and self.current_round <= self.last_announced_round + 1
        ):
            return

        text = (
            f"E|{self.current_round}|{target_id}|"
            f"{enemy['position']['q']}|{enemy['position']['r']}"
        )

        here = me.position(observation)

        report_allies = sorted(
            visible.allies(observation),
            key=lambda ally: (
                tile.distance(
                    here,
                    ally["position"],
                ),
                ally["unit_id"],
            ),
        )[:4]

        for ally in report_allies:

            player = self.ally_player_by_unit_id.get(
                ally["unit_id"]
            )

            if player is None:
                continue

            self.pending_messages.append(
                {
                    "to": player,
                    "text": text,
                }
            )

        self.last_announced_target = target_id
        self.last_announced_round = self.current_round

    def _shared_enemy_is_fresh(
        self,
        max_age: int = 3,
    ) -> bool:

        if self.shared_enemy_position is None:
            return False

        return (
            self.current_round
            - self.shared_enemy_round
            <= max_age
        )

    # ============================================================
    # SEASON 5 HOME CAPTURE DEFENSE
    # ============================================================

    def _home_capture_zone(
        self,
        observation: SkirmishObservation,
    ):
        """
        Return the capture zone on OUR side of the battlefield.

        The three Season 5 zones are spread between the two armies.
        Determine our side from me.direction(), build a home-side anchor,
        and select the zone nearest that anchor.
        """

        areas = list(
            zone.zones(observation)
        )

        if not areas:
            return None

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

        backward = opposite[
            forward
        ]

        anchor = {
            "q": center["q"],
            "r": center["r"],
        }

        # Move the anchor deep toward our own side.
        for _ in range(
            max(
                3,
                self._field_extent(
                    observation
                ) - 2,
            )
        ):
            anchor = tile.neighbors(
                anchor
            )[backward]

        return min(
            areas,
            key=lambda area: tile.distance(
                area["center"],
                anchor,
            ),
        )

    def _middle_capture_zone(
        self,
        observation: SkirmishObservation,
    ):
        """
        Return the capture zone closest to the battlefield center.

        With Season 5's three spread-out zones, this is the contested
        middle objective that the mobile footmen should reinforce.
        """

        areas = list(
            zone.zones(observation)
        )

        if not areas:
            return None

        center = tile.at_center(
            observation
        )

        return min(
            areas,
            key=lambda area: tile.distance(
                area["center"],
                center,
            ),
        )


    def _select_home_defender(
        self,
        observation: SkirmishObservation,
    ) -> str | None:
        """
        Select the footman whose STARTING position is closest to our
        home capture zone.

        At army spawn the nearby friendly footmen are visible together,
        allowing each Agent instance to make the same deterministic
        selection. Unit ID breaks distance ties.
        """

        area = self._home_capture_zone(
            observation
        )

        if area is None:
            return None

        candidates = []

        if me.unit_type(observation) == "footman":

            candidates.append(
                {
                    "unit_id": me.unit_id(
                        observation
                    ),
                    "position": me.position(
                        observation
                    ),
                }
            )

        for ally in visible.allies(
            observation
        ):

            if ally["type"] != "footman":
                continue

            candidates.append(
                {
                    "unit_id": ally[
                        "unit_id"
                    ],
                    "position": ally[
                        "position"
                    ],
                }
            )

        if not candidates:
            return None

        chosen = min(
            candidates,
            key=lambda unit: (
                tile.distance(
                    unit["position"],
                    area["center"],
                ),
                unit["unit_id"],
            ),
        )

        return chosen[
            "unit_id"
        ]

    def _act_home_defender(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:
        """
        Move onto our home capture point and HOLD it.

        The defender will attack an enemy already within footman attack
        range, but it will not chase enemies away from the objective.
        """

        area = self._home_capture_zone(
            observation
        )

        if area is None:
            return action.stay()

        here = me.position(
            observation
        )

        enemies = visible.enemies(
            observation
        )

        attack_range = units.STATS[
            "footman"
        ].attack_range

        attackable = [
            enemy
            for enemy in enemies
            if tile.distance(
                here,
                enemy["position"],
            ) <= attack_range
        ]

        if attackable:

            target = self._choose_target(
                attackable
            )

            return action.stay(
                target["unit_id"],
                observation,
            )

        # Once we are physically inside our zone, stay there.
        if zone.at(
            observation,
            here,
        ) is not None:

            # zone.at() may refer to one of the other zones, so confirm
            # we are actually near our designated home zone.
            if tile.distance(
                here,
                area["center"],
            ) <= 2:
                return action.stay()

        path = self._path_toward(
            observation,
            area["center"],
            max_steps=2,
            allow_close_detour=True,
        )

        if path != 0:
            return action.move(path)

        return action.stay()

    # ============================================================
    # TARGET SELECTION
    # ============================================================

    def _choose_target(
        self,
        enemies: list[dict],
    ) -> dict:
        """
        Shared deterministic focus-fire rule.

        If a squadmate recently named a target and that target is
        visible now, honor the communicated target first. Otherwise
        choose the lowest-HP enemy, then unit ID for a stable tie.
        """

        if self.shared_target_id is not None:
            for enemy in enemies:
                if enemy["unit_id"] == self.shared_target_id:
                    return enemy

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
        Improve an archer's firing position without sacrificing
        firing uptime.

        Only legal endpoints that keep the target in attack range
        are considered. A hill firing downhill is strongly favored,
        but the archer will not move just for terrain unless the new
        position is meaningfully better.
        """

        here = me.position(observation)
        allies = self._visible_squadmates(observation)

        attack_range = units.STATS[
            "archer"
        ].attack_range

        # If somehow the current tile no longer has a shot, terrain
        # repositioning is not the job of this helper.
        if tile.distance(
            here,
            target_position,
        ) > attack_range:
            return 0

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

            directions = paths.decode(path)

            if len(directions) > 2:
                continue

            end = tile.at_path_end(
                here,
                path,
            )

            if tile.distance(
                end,
                target_position,
            ) > attack_range:
                continue

            score = self._archer_position_score(
                observation,
                end,
                target_position,
                allies,
            )

            # Avoid pointless shuffling. Reposition only when the
            # new firing tile is clearly better.
            if score > best_score + 5:
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

        target_info = tile.terrain_at(
            observation,
            target_position,
        )

        terrain_type = info["terrain"]
        feature = info["feature"]
        target_terrain = target_info["terrain"]

        if terrain_type in (
            "water",
            "void",
        ):
            return -10000

        score = 0

        # --------------------------------------------------------
        # DAMAGE ADVANTAGE
        #
        # Season terrain gives attacks from hill -> lower ground
        # +1 damage. Make this the archer's dominant positional goal.
        # --------------------------------------------------------

        if terrain_type == "hill":

            # Hill shooting downhill is the ideal firing position.
            if target_terrain != "hill":
                score += 60
            else:
                score += 22

        # Forest is still useful defensively, but it should not
        # outweigh a real hill damage advantage.
        if feature == "forest":
            score += 10

        if feature == "marsh":
            score -= 12

        # --------------------------------------------------------
        # ALLIED SUPPORT
        # --------------------------------------------------------

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
                score += 6

            elif nearest_ally <= 6:
                score += 1

            else:
                score -= 10

        # --------------------------------------------------------
        # USE ARCHER RANGE
        # --------------------------------------------------------

        target_distance = tile.distance(
            position,
            target_position,
        )

        if target_distance >= 5:
            score += 10

        elif target_distance >= 3:
            score += 5

        elif target_distance <= 1:
            score -= 18

        elif target_distance == 2:
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

    def _choose_cavalry_target(
        self,
        observation: SkirmishObservation,
        enemies: list[dict],
    ) -> dict:
        """
        Cavalry target priority is based on WHAT CAN BE HIT THIS TURN.

        Order:
        1. Enemy already in attack range.
        2. Enemy cavalry can reach and attack this activation.
           Fewer movement steps is better.
        3. Otherwise, nearest visible enemy.
        4. Only after reachability/distance do HP and unit type matter.

        This intentionally ignores shared_target_id. A communicated archer
        across the battlefield must never outrank a footman/cavalry that
        can actually be struck right now.
        """

        here = me.position(
            observation
        )

        attack_range = units.STATS[
            "cavalry"
        ].attack_range

        legal_paths = [
            path
            for path in action.legal_paths(
                observation
            )
            if path != 0
        ]

        scored = []

        for enemy in enemies:

            current_distance = tile.distance(
                here,
                enemy["position"],
            )

            # Zero steps means cavalry can attack immediately.
            if current_distance <= attack_range:

                min_attack_steps = 0

            else:

                min_attack_steps = None

                for path in legal_paths:

                    end = tile.at_path_end(
                        here,
                        path,
                    )

                    if tile.distance(
                        end,
                        enemy["position"],
                    ) > attack_range:
                        continue

                    step_count = len(
                        paths.decode(path)
                    )

                    if (
                        min_attack_steps is None
                        or step_count
                        < min_attack_steps
                    ):
                        min_attack_steps = (
                            step_count
                        )

            reachable_this_turn = (
                min_attack_steps
                is not None
            )

            scored.append(
                (
                    # Reachable attack ALWAYS beats unreachable target.
                    0 if reachable_this_turn else 1,

                    # Among reachable targets, hit the one requiring
                    # the least movement.
                    (
                        min_attack_steps
                        if reachable_this_turn
                        else current_distance
                    ),

                    # Next prefer physical proximity.
                    current_distance,

                    # Then prefer a weaker target.
                    enemy["hit_points"],

                    # Archer preference is only a late tie-breaker.
                    0 if enemy["type"] == "archer" else 1,

                    enemy["unit_id"],

                    enemy,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
                item[4],
                item[5],
            )
        )

        return scored[0][-1]


    def _cavalry_attack_path(
        self,
        observation: SkirmishObservation,
        target_position: AxialPosition,
    ) -> int:
        """
        Find a supported attack route.

        Priorities:
        1. End within cavalry attack range.
        2. If possible, move 3+ hexes so Charge activates.
        3. Otherwise make real progress toward the target.

        Staying and attacking is handled BEFORE this helper is called.
        """

        here = me.position(
            observation
        )

        attack_range = units.STATS[
            "cavalry"
        ].attack_range

        best_path = 0
        best_score = -100000

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

            progress = (
                start_distance
                - target_distance
            )

            score = (
                progress * 40
                - target_distance * 20
            )

            # Actually getting into attack range is far more important
            # than merely moving a long distance.
            if target_distance <= attack_range:
                score += 300

                if displacement >= 3:
                    score += 120

            elif displacement >= 3:
                # Charge-shaped movement with no attack is only mildly
                # useful, not a reason to abandon the group.
                score += 15

            score += self._terrain_score(
                observation,
                end,
            )

            if target_distance > start_distance:
                score -= 100

            if score > best_score:

                best_score = score
                best_path = path

        return best_path

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
        allow_close_detour: bool = False,
    ) -> int:
        """
        Move toward a goal while avoiding short movement loops.

        The old version separated all "distance improving" paths from
        detours and ALWAYS chose an improving path first. That meant a
        recently visited hex one step closer to the goal could beat a
        necessary sideways route around allied units.

        This version scores all legal endpoints together. Forward
        progress is still strongly preferred, but recently visited hexes
        receive enough penalty for a real detour to win.
        """

        here = me.position(observation)

        current_distance = tile.distance(
            here,
            goal,
        )

        candidates = []

        recent = set(
            self.recent_positions
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

            end = tile.at_path_end(
                here,
                path,
            )

            distance = tile.distance(
                end,
                goal,
            )

            progress = (
                current_distance
                - distance
            )

            terrain_score = self._terrain_score(
                observation,
                end,
            )

            path_length = len(
                directions
            )

            score = (
                progress * 120
                - distance * 12
                + terrain_score * 3
                - path_length
            )

            end_key = (
                end["q"],
                end["r"],
            )

            # A recently visited hex is the clearest indication that we
            # are cycling around a blocker. This penalty is intentionally
            # larger than the value of one hex of forward progress.
            if end_key in recent:
                score -= 220

            # The immediately previous hex is especially undesirable.
            if self.avoid_position is not None:
                if (
                    end["q"]
                    == self.avoid_position["q"]
                    and end["r"]
                    == self.avoid_position["r"]
                ):
                    score -= 120

            # Ordinary formation movement should not wander once already
            # next to its destination. Combat/capture navigation may need
            # a temporary sideways step around occupied hexes.
            if (
                current_distance <= 2
                and progress <= 0
                and not allow_close_detour
            ):
                continue

            candidates.append(
                (
                    score,
                    progress,
                    path,
                )
            )

        if not candidates:
            return 0

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return candidates[0][2]

    def _unstick_path(
        self,
        observation: SkirmishObservation,
        max_steps: int | None = None,
    ) -> int:
        """
        Choose a legal escape route when normal goal-directed movement
        has caused repeated passes.

        Preference:
        - move toward the enemy half / center,
        - avoid immediately returning to the previous hex,
        - avoid poor terrain,
        - create some separation from nearby allied traffic.
        """

        here = me.position(observation)
        center = tile.at_center(observation)

        allies = visible.allies(observation)

        best_path = 0
        best_score = -100000

        for path in action.legal_paths(observation):

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

            score = 0

            # General forward pressure.
            score -= tile.distance(
                end,
                center,
            ) * 8

            score += self._terrain_score(
                observation,
                end,
            ) * 2

            # Strongly avoid bouncing through recently visited hexes.
            if (
                end["q"],
                end["r"],
            ) in set(self.recent_positions):
                score -= 180

            if self.avoid_position is not None:
                if (
                    end["q"] == self.avoid_position["q"]
                    and end["r"] == self.avoid_position["r"]
                ):
                    score -= 100

            # Prefer a tile with a little breathing room when allied
            # traffic is causing the blockage.
            if allies:
                nearest_ally = min(
                    tile.distance(
                        end,
                        ally["position"],
                    )
                    for ally in allies
                )
                score += min(nearest_ally, 3) * 5

            # Longer cavalry escapes are useful, while max_steps keeps
            # footmen/archers within their normal tactical movement.
            score += len(directions)

            if score > best_score:
                best_score = score
                best_path = path

        return best_path

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
