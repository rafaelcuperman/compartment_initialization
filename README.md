This is the code for the prophylactic dose and time prediction algorithm.
The main code is in `src/dose_time_prediction.jl`.

An example on how to use the code on a single individual can be found in `scripts/dose_time_prediction.jl`.

# How to use this code

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> compartment-initialization

To (locally) reproduce this project, do the following:

0. Download this code base.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "compartment-initialization"
```
which auto-activate the project and enable local path handling from DrWatson.


