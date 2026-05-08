# HOT-Step Community Plugins

Drop custom Lua plugin files here to extend the engine without rebuilding.

## Directory Structure

```
plugins/
├── solvers/        ← Custom ODE/SDE solvers
├── schedulers/     ← Custom noise schedules
└── guidance/       ← Custom guidance modes
```

## How It Works

1. Place `.lua` files in the appropriate subdirectory
2. Restart the engine (or the app)
3. Your plugin appears in the UI dropdown automatically

The engine scans `engine/plugins/` (built-in) first, then this `plugins/` directory.
Duplicate names are skipped with a console warning.

## Writing a Plugin

Every plugin is a single `.lua` file that returns a table with metadata and a `step()` function.

### Solver Example

```lua
return {
  name    = "my_solver",
  display = "My Custom Solver",
  type    = "solver",
  nfe     = 1,
  accent  = "pink",

  -- Optional user-facing parameters
  params = {
    { key = "strength", type = "slider", label = "Strength",
      default = 0.5, min = 0, max = 1, step = 0.01 },
  },

  step = function(x, v, t, t_next, dt, params)
    -- x: current latent (FloatArray)
    -- v: velocity prediction (FloatArray)
    -- t, t_next, dt: timestep scalars
    -- params: table of user values { strength = "0.5", ... }
    for i = 0, x:size() - 1 do
      x:set(i, x:get(i) + dt * v:get(i))
    end
  end,
}
```

### Scheduler Example

```lua
return {
  name    = "my_schedule",
  display = "My Schedule",
  type    = "scheduler",

  schedule = function(n_steps, params)
    -- Return a table of n_steps+1 descending floats from 1.0 to 0.0
    local ts = {}
    for i = 0, n_steps do
      ts[i + 1] = 1.0 - i / n_steps
    end
    return ts
  end,
}
```

### Guidance Example

```lua
return {
  name    = "my_guidance",
  display = "My Guidance",
  type    = "guidance",

  guide = function(cond, uncond, scale, t, params)
    -- cond/uncond: FloatArray (conditional/unconditional predictions)
    -- scale: guidance scale (number)
    -- t: current timestep (0→1)
    -- Return guided prediction in cond (modified in-place)
    for i = 0, cond:size() - 1 do
      local c = cond:get(i)
      local u = uncond:get(i)
      cond:set(i, u + scale * (c - u))
    end
  end,
}
```

## Parameter Types

| Type     | Fields                                           |
|----------|--------------------------------------------------|
| `slider` | `key`, `label`, `default`, `min`, `max`, `step`  |
| `select` | `key`, `label`, `default`, `options`              |
| `toggle` | `key`, `label`, `default`                        |
| `text`   | `key`, `label`, `default`, `hint`                |

## Safety

Plugins run in a sandboxed Lua environment:
- ❌ No `os`, `io`, `debug`, `dofile`, `loadfile`
- ✅ `math`, `string`, `table`, `require` (for companion data files)
- ✅ Full `FloatArray` API for zero-copy memory access

## Sharing Plugins

Share your `.lua` files with other HOT-Step users! Just drop them in the right folder.
