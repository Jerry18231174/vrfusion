import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

class NormalIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.background = props.get("background", mi.Color3f(0.1))

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        # Render normal map

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        result = mi.Color3f(0.0)
        result[si.is_valid()] = dr.fma(si.n, 0.5, 0.5)
        result[~si.is_valid()] = self.background

        return result, si.is_valid(), []

mi.register_integrator("normal", lambda props: NormalIntegrator(props))
