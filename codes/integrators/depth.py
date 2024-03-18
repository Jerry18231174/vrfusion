import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

class DepthIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.color = props.get("color", mi.Color3f(0.2, 0.5, 0.2))
        self.background = props.get("background", mi.Color3f(0.1))

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        # Compute max depth of the scene

        camera: mi.ProjectiveCamera = scene.sensors()[0]
        camera_pos: mi.Point3f = camera.world_transform().translation()
        box: mi.BoundingBox3f = scene.bbox()

        max_depth: mi.Float = mi.Float(0.0)
        for i in range(8):
            max_depth = dr.maximum(
                max_depth,
                dr.norm(box.corner(i) - camera_pos)
            )

        # Render depth map

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        result = mi.Color3f(0.0)
        if True or self.fusion_mode:
            dist = dr.norm(si.p - camera_pos)
            result[si.is_valid()] = dist
            result[~si.is_valid()] = 100000.0
        else:
            dist = dr.norm(si.p - camera_pos) / max_depth
            result[si.is_valid()] = self.color * (1.0 - dist)
            result[~si.is_valid()] = self.background

        return result, si.is_valid(), []

mi.register_integrator("depth", lambda props: DepthIntegrator(props))
