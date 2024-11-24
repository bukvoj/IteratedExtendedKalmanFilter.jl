module IteratedExtendedKalmanFilter
    # Includes
    using Enzyme
    using LinearAlgebra

    # Source files
    include("predict.jl")
    include("correct.jl")

    # Export
    export  iekfpredict,
            iekfcorrect

end
