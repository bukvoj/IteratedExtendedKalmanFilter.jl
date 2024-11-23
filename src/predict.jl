function iekfpredict(f::Function, F::AbstractArray, x::Vector,P::AbstractArray,Q::AbstractArray,u...)
    x_new = f(x,u...)
    P_new = F*P*F' + Q
    return x_new,P_new
end

# function iekfpredict(f::Function, F::AbstractArray, x::Vector,P::AbstractArray,Q::AbstractArray)
#     x_new = f(x)
#     P_new = F*P*F' + Q
#     return x_new,P_new
# end

function iekfpredict(f::Function, getjac::Function, x::Vector,P::AbstractArray,Q::AbstractArray,u...)
    F = getjac(x,u...)
    return iekfpredict(f,F,x,P,Q,u...)
end

# function iekfpredict(f::Function, getjac::Function, x::Vector,P::AbstractArray,Q::AbstractArray)
#     F = getjac(x)
#     return iekfpredict(f,F,x,P,Q)    
# end

function iekfpredict(f::Function, x::Vector,P::AbstractArray,Q::AbstractArray,u...)
    ff = args -> f(args,u...)
    F,x_new = jacobian(ForwardWithPrimal,ff,x)
    P_new = F[1]*P*F[1]' + Q
    return x_new,P_new
end

# function iekfpredict(f::Function, x::Vector,P::AbstractArray,Q::AbstractArray)
#     F,x_new = jacobian(ForwardWithPrimal,f,x)
#     P_new = F[1]*P*F[1]' + Q
#     return x_new,P_new
# end


