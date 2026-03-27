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
