# Implements the data (measurement, correction) step of the iekf algorithm

function iekfcorrect(h::Function, 
                    getjac::Function, 
                    x::Union{Vector, Real},
                    P::Union{AbstractArray, Real},
                    y::Union{Vector, Real},
                    R::Union{AbstractArray, Real},
                    maxiters=30,eps=1e-8,
                    u...)
    xi = x
    i = 1
    while true
        prev = xi
        Hi = getjac(x,u...)
        residual =  y - h(x,u...)
        Si = Hi*P*Hi' + R
        Ki = P*Hi'*inv(Si)
        xi = x + Ki*(residual-Hi*(x-xi))
        if norm(xi-prev) < eps
            return xi, P-Ki*Hi*P, residual, Si, Ki, Hi, i
        elseif i == maxiters
            return xi, P-Ki*Hi*P, residual, Si, Ki, Hi, i
        else
            i += 1
        end
    end
end

function iekfcorrect(h::Function, 
                    x::Union{Vector, Real},
                    P::Union{AbstractArray, Real},
                    y::Union{Vector, Real},
                    R::Union{AbstractArray, Real},
                    maxiters=30,eps=1e-8,
                    u...)
    xi = x/1 # This is a hack to make the type system happy
    i = 1
    while true
        prev = xi
        hh = args -> h(args,u...)
        tmp,y_hat = jacobian(ForwardWithPrimal, hh, xi)
        Hi = tmp[1]
        residual =  y - y_hat
        Si = Hi*P*Hi' + R
        Ki = P*Hi'*inv(Si)
        xi = x + Ki*(residual-Hi*(x-xi))
        if norm(xi-prev) < eps
            return xi, P-Ki*Hi*P, residual, Si, Ki, Hi, i
        elseif i == maxiters
            return xi, P-Ki*Hi*P, residual, Si, Ki, Hi, i
        else
            i += 1
        end
    end
end