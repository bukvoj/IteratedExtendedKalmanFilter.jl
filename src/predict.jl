function iekfpredict(f::Function, 
                    F::AbstractArray, 
                    x::Union{Vector, Real},
                    P::Union{AbstractArray, Real},
                    Q::Union{AbstractArray,Real},
                    u...)
    x_new = f(x,u...)
    P_new = F*P*F' + Q
    return x_new,P_new
end

function iekfpredict(f::Function, 
                    getjac::Function, 
                    x::Union{Vector, Real},
                    P::Union{AbstractArray, Real},
                    Q::Union{AbstractArray,Real},
                    u...)
    F = getjac(x,u...)
    x_new = f(x,u...)
    P_new = F*P*F' + Q
    return x_new,P_new
end

function iekfpredict(f::Function, 
                    x::Union{Vector, Real},
                    P::Union{AbstractArray, Real},
                    Q::Union{AbstractArray,Real},
                    u...)
    ff = args -> f(args,u...)
    F,x_new = jacobian(ForwardWithPrimal,ff,x/1) # /1 is a hack to make the type system happy
    P_new = F[1]*P*F[1]' + Q
    return x_new,P_new
end


