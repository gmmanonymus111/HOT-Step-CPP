-- cfg_mp.lua: CFG-MP — Manifold Projection Guidance
-- Paper: "Improving CFG of Flow Matching via Manifold Projection"
--        Su et al., 2025 (arXiv:2601.21892)
--
-- After each solver step, projects the latent back onto a manifold where the
-- prediction gap (cond - uncond) is minimised. Uses iterative fixed-point
-- iteration of the operator G(x, t):
--
--   z = x - a * v_uncond(t, x)      -- push away from unconditioned manifold
--   x = z + a * v_cond(t, z)        -- pull toward conditioned manifold
--
-- where a = |dt| / 2 (validated in paper Appendix C.2.1).
--
-- The guide() function applies standard linear CFG for the base velocity.
-- The post_step() function performs K iterations of manifold projection using
-- real model evaluations at the post-solver latent position.
--
-- Performance note: each iteration = 2 extra NFEs (one cond, one uncond).
-- K=2 adds ~3x total compute; K=1 adds ~2x.

guidance = {
    name        = "cfg_mp",
    display     = "CFG-MP",
    description = "Manifold projection guidance (Su et al. 2025)",
    params      = {
        { key = "iterations", type = "slider", label = "Projection Iterations (K)",
          default = 1, min = 1, max = 5, step = 1,
          hint = "Fixed-point iterations per step. Paper recommends 2." },
    },
}

-- Standard linear CFG for the base velocity step
function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    -- Route through native APG for momentum/projection consistency
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
end

-- Post-step manifold projection: called AFTER the solver updates xt
-- Args:
--   xt           : mutable FloatArray — current latent state (modified in-place)
--   t            : float — timestep (t_next, the timestep we just stepped TO)
--   n            : int — total elements in xt
--   eval_cond    : function(xt_arr, t) — evaluates model with conditioning → vt_cond
--   eval_uncond  : function(xt_arr, t) — evaluates model without conditioning → vt_uncond
--   vt_cond      : mutable FloatArray — output buffer for conditional velocity
--   vt_uncond    : mutable FloatArray — output buffer for unconditional velocity
function post_step(xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond)
    local K = (params and params.iterations) or 2
    local a = math.abs(dt or 0.03) / 2.0  -- dt is a global from the C++ bridge

    for k = 1, K do
        -- Step 1: z = xt - a * v_uncond(t, xt)
        eval_uncond(xt, t)
        for i = 0, n - 1 do
            xt[i] = xt[i] - a * vt_uncond[i]
        end

        -- Step 2: xt = z + a * v_cond(t, z)
        eval_cond(xt, t)
        for i = 0, n - 1 do
            xt[i] = xt[i] + a * vt_cond[i]
        end
    end
end
