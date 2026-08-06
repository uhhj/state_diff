"""R3 coupon and table-settle variants using composite edge-angle material."""
from state_diff.env.block_pushing.angle_elastic_soft_block import (
    AngleElasticSoftBlock, load_angle_material_profile)
from state_diff.env.block_pushing.material_coupon_r2 import MaterialCouponR2
from state_diff.env.block_pushing.table_settle_coupon import TableSettleCoupon


class _AngleElasticHooks:
    def _load_profile(self, path: str):
        material, self.angle_parameters, payload = load_angle_material_profile(path)
        return material, payload

    def _create_angle_block(self, block_config, soft, static_indices=None):
        return AngleElasticSoftBlock(
            self.client, block_config, self.material, self.angle_parameters,
            float(soft["spring_force_cap_n"]), tuple(soft["center_xy"]),
            float(soft["yaw_deg"]), static_indices=static_indices)

    def _energy_fields(self, stats) -> dict:
        result = super()._energy_fields(stats)
        result["spring_energy_angle_j"] = stats.final_energy_by_kind_j["angle"]
        return result


class AngleElasticMaterialCoupon(_AngleElasticHooks, MaterialCouponR2):
    """Static-face free-space coupon with R3 angle elasticity."""

    def _create_block(self, block_config, soft: dict):
        return self._create_angle_block(
            block_config, soft, self.anchor_indices.tolist())


class AngleElasticTableSettleCoupon(_AngleElasticHooks, TableSettleCoupon):
    """Gravity table-settle coupon with R3 angle elasticity."""

    def _create_block(self, block_config, soft: dict):
        return self._create_angle_block(block_config, soft)
