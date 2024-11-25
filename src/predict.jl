# This file contains the predict step of the EKF and IEKF algorithms.

# This is to differentiate between scalar and multidimensional system
for T in ((Vector{<:Real},AbstractArray{<:Real}),(Real,Real)) 
@eval begin

function iekfpredict(f::Function, 
                    F::AbstractArray, 
                    x::$T[1],
                    P::$T[2],
                    Q::$T[2],
                    u...)
    x_new = f(x,u...)
    P_new = F*P*F' + Q
    return x_new,P_new
end

function iekfpredict(f::Function, 
                    getjac::Function, 
                    x::Union{Vector, Real},
                    P::$T[2],
                    Q::$T[2],
                    u...)
    F = getjac(x,u...)
    x_new = f(x,u...)
    P_new = F*P*F' + Q
    return x_new,P_new
end

function iekfpredict(f::Function, 
                    x::Union{Vector, Real},
                    P::$T[2],
                    Q::$T[2],
                    u...)
    ff = args -> f(args,u...)
    F,x_new = jacobian(ForwardWithPrimal,ff,x/1) # /1 is a hack to make the type system happy
    P_new = F[1]*P*F[1]' + Q
    return x_new,P_new
end

end #eval
end #for

