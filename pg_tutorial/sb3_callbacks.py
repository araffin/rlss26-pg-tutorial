from stable_baselines3.common.callbacks import BaseCallback


class RacingInfoCallback(BaseCallback):
    """Log extra racing metrics (lap count, progress, slip) to TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            # if "lap_count" in info:
            #     self.logger.record("racing/lap_count", info["lap_count"])
            if "best_lap_time" in info:
                self.logger.record("racing/best_lap_time", info["best_lap_time"])
            if "last_lap_time" in info:
                self.logger.record("racing/last_lap_time", info["last_lap_time"])
        return True


class PongInfoCallback(BaseCallback):
    """Log Pong match metrics from env info dictionaries to TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            sb3_logs = info.get("sb3_logs")
            if isinstance(sb3_logs, dict):
                for key, value in sb3_logs.items():
                    self.logger.record(key, value)

            if info.get("match_over"):
                self.logger.record("pong/final_score_left", info.get("score_left", 0))
                self.logger.record("pong/final_score_right", info.get("score_right", 0))
                winner = info.get("winner")
                self.logger.record("pong/final_agent_win", float(winner == "left"))
                self.logger.record("pong/final_opponent_win", float(winner == "right"))
        return True
