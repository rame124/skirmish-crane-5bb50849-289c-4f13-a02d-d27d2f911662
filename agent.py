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
    - Cavalry prefers charge attacks when fighting with its squad.
    - Half-health cavalry regroups, but may rejoin combat once
      friendly support is present.
    - Movement can detour around rivers and other obstacles instead
      of freezing when no legal path directly reduces distance.
    - Units dynamically cooperate with nearby allies and communicate enemy targets/last-known positions.
    - Reserve footmen prioritize the capture zone while cooperating with nearby allies.
    """

    def reset(self, seed, observation) -> None:
        self.state = State.ASSEMBLE

        # Used to discourage immediate backtracking while navigating
        # around rivers and other obstacles.
        self.previous_position = None
        self.avoid_position = None

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
        # Healthy cavalry is intentionally aggressive. Its charge
        # ability rewards long movement, and scouting contact is
        # valuable even if the cavalry exposes itself.
        # --------------------------------------------------------

        if unit_type == "cavalry":

            hp = me.hit_points(observation)
            max_hp = units.STATS["cavalry"].hit_points

            if hp <= max_hp / 2:

                # Wounded cavalry only regroups if it is isolated.
                if enemies and nearby_mates:
                    self.state = State.ENGAGE
                else:
                    self.state = State.REGROUP

                return

            if enemies:
                self.state = State.ENGAGE
                return

            # Healthy cavalry should keep advancing rather than
            # waiting for a local squad to form.
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
                # Footmen can commit alone if necessary.
                self.state = State.ENGAGE

            else:
                # Lone archers should avoid close unsupported combat,
                # but their action logic may still move toward a target
                # to get into firing range.
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

        # ADVANCE stays ADVANCE when there is no contact.
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

        enemies = visible.enemies(observation)

        area = self._capture_zone(
            observation,
        )

        zone_needs_help = (
            area is not None
            and self._capture_needs_reinforcement(
                observation,
                desired_allies=3,
            )
        )

        # --------------------------------------------------------
        # ACTIVE COMBAT
        #
        # If the zone still needs bodies, footmen only break off for
        # enemies that are near enough to directly contest the route
        # or the capture point.
        # --------------------------------------------------------

        if enemies:

            target = self._choose_target(
                enemies,
            )

            if zone_needs_help:

                capture_center = area["center"]

                enemy_near_capture = tile.distance(
                    target["position"],
                    capture_center,
                ) <= 4

                enemy_near_self = tile.distance(
                    me.position(observation),
                    target["position"],
                ) <= 2

                if not (
                    enemy_near_capture
                    or enemy_near_self
                ):

                    path = self._path_toward(
                        observation,
                        capture_center,
                        max_steps=2,
                        allow_close_detour=True,
                    )

                    if path != 0:
                        return action.move(
                            path,
                            target["unit_id"],
                            observation,
                        )

            # Either the zone is sufficiently occupied, or the enemy
            # is an immediate threat to us/the objective.
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
        # Reinforce capture until at least three visible friendlies
        # occupy the zone. After that, resume normal aggression.
        # --------------------------------------------------------

        if zone_needs_help:

            path = self._path_toward(
                observation,
                area["center"],
                max_steps=2,
                allow_close_detour=True,
            )

            if path != 0:
                return action.move(path)

            return action.stay()

        goal = self._advance_goal(
            observation,
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
        # ENEMY VISIBLE
        #
        # Healthy cavalry attacks regardless of support. Prefer a
        # long displacement so Charge triggers whenever practical.
        # --------------------------------------------------------

        if enemies:

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

            # If the charge planner cannot find anything useful,
            # still close aggressively instead of waiting.
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

            return action.stay(
                target["unit_id"],
                observation,
            )

        # --------------------------------------------------------
        # WOUNDED AND NO CONTACT
        #
        # Half-health cavalry may regroup.
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
        # HEALTHY SCOUT / CHARGE FORWARD
        #
        # Use full cavalry movement. Shared enemy intel is preferred,
        # otherwise advance toward the enemy side. Do NOT slow to
        # footman/archer speed.
        # --------------------------------------------------------

        if self._shared_enemy_is_fresh():

            goal = {
                "q": self.shared_enemy_position["q"],
                "r": self.shared_enemy_position["r"],
            }

        else:

            # Push toward the mirrored friendly rally point on the
            # enemy half. This maintains broad forward pressure.
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
        Receive delayed squad reports and send reports queued by act().

        Messages deliberately stay compact:
            E|round|unit_id|q|r

        A message cannot affect the order chosen on the same activation;
        it is stored for the recipient's next activation.
        """

        for message in inbox:
            text = str(message.get("text", ""))

            if not text.startswith("E|"):
                continue

            parts = text.split("|")

            if len(parts) != 5:
                continue

            try:
                report_round = int(parts[1])
                unit_id = parts[2]
                q = int(parts[3])
                r = int(parts[4])
            except (TypeError, ValueError):
                continue

            # Keep only the newest report received.
            if report_round >= self.shared_enemy_round:
                self.shared_target_id = unit_id
                self.shared_enemy_position = {
                    "q": q,
                    "r": r,
                }
                self.shared_enemy_round = report_round

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
    # CAPTURE ZONE
    # ============================================================

    def _capture_zone(
        self,
        observation: SkirmishObservation,
    ):

        areas = zone.zones(observation)

        if not areas:
            return None

        # Season 4 has one capture zone.
        return areas[0]

    def _capture_center(
        self,
        observation: SkirmishObservation,
    ) -> AxialPosition:

        area = self._capture_zone(observation)

        if area is None:
            return tile.at_center(observation)

        return area["center"]

    def _is_capture_reserve(
        self,
        observation: SkirmishObservation,
    ) -> bool:

        return (
            me.unit_type(observation) == "footman"
            and self._unit_index(observation) >= 6
        )

    def _capture_needs_reinforcement(
        self,
        observation: SkirmishObservation,
        desired_allies: int = 3,
    ) -> bool:
        """
        Return True until the capture zone visibly contains enough
        friendly troops.

        zone.occupants() is visibility-limited, but for this purpose
        that is acceptable: if this unit cannot confirm that the zone
        is reinforced, sending another footman is the safer choice.
        """

        area = self._capture_zone(
            observation,
        )

        if area is None:
            return False

        occupants = zone.occupants(
            observation,
            area,
        )

        own_side = me.side(
            observation,
        )

        friendly_count = sum(
            1
            for unit in occupants
            if unit.get("side") == own_side
        )

        return friendly_count < desired_allies


    def _act_capture_footman(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:
        """
        Reserve footmen occupy the capture point and try to remain
        adjacent so Season 4's Shield Wall ability can protect them.
        """

        area = self._capture_zone(observation)

        if area is None:
            goal = tile.at_center(observation)
        else:
            goal = area["center"]

        here = me.position(observation)
        mates = self._visible_squadmates(observation)

        # Once inside the zone, staying adjacent to the reserve partner
        # is more valuable than endlessly jockeying for the center tile.
        in_zone = (
            area is not None
            and zone.at(observation, here) is not None
        )

        if mates:
            partner = min(
                mates,
                key=lambda ally: tile.distance(
                    here,
                    ally["position"],
                ),
            )

            partner_distance = tile.distance(
                here,
                partner["position"],
            )

            if in_zone and partner_distance <= 1:
                return action.stay()

            if partner_distance > 1:
                path = self._path_toward(
                    observation,
                    partner["position"],
                    max_steps=2,
                )

                if path != 0:
                    return action.move(path)

        path = self._path_toward(
            observation,
            goal,
            max_steps=2,
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
        allow_close_detour: bool = False,
    ) -> int:
        """
        Move toward the goal when possible.

        If no legal path gets closer, allow a sideways/detour move.
        allow_close_detour=True is used for melee units near rivers or
        traffic jams, where being only one or two hexes away does NOT
        mean the target is actually reachable.
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

        if improving_paths:

            improving_paths.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            return improving_paths[0][1]

        # For ordinary formation movement, don't wander around an
        # occupied nearby goal. For active melee/capture navigation,
        # however, allow detours around rivers and crowded allies.
        if (
            current_distance <= 2
            and not allow_close_detour
        ):
            return 0

        if detour_paths:

            detour_paths.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            return detour_paths[0][1]

        return 0

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

            # Strongly avoid bouncing to the last hex.
            if self.avoid_position is not None:
                if (
                    end["q"] == self.avoid_position["q"]
                    and end["r"] == self.avoid_position["r"]
                ):
                    score -= 80

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
