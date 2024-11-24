function powers(x::Vector)
    for i in 1:length(x)
        x[i] = x[i]^i
    end
    x
end

function upowers1(x::Vector,u)
    for i in 1:length(x)
        x[i] = u*x[i]^i 
    end
    x
end

function upowers2(x::Vector,u)
    for i in 1:length(x)
        x[i] = u[i]*x[i]^i
    end
    x
end

function powersjac(x)
    J = zeros(length(x),length(x))
    for i in 1:length(x)
        J[i,i] = i*x[i]^(i-1)
    end
    J
end

function upowersjac1(x,u)
    J = zeros(length(x),length(x))
    for i in 1:length(x)
        J[i,i] = i*u*x[i]^(i-1)
    end
    J
end

function upowersjac2(x,u)
    J = zeros(length(x),length(x))
    for i in 1:length(x)
        J[i,i] = i*u[i]*x[i]^(i-1)
    end
    J
end

F1 = powersjac([1.0,2.0,3.0])
F2 = upowersjac1([1.0,2.0,3.0],2.3)
F3 = upowersjac2([1.0,2.0,3.0],[1,2,3])

@testset "predict.jl: P=0" begin
    Q = rand(3,3)
    Q = Q+Q' # Make Q symmetric
    # Autodiff tests
    x,P = iekfpredict(powers,[1.0,2.0,3.0],zeros(3,3),Q)
    @test P == Q
    @test x == [1.0,4.0,27.0]
    x,P = iekfpredict(upowers1,[1.0,2.0,3.0],zeros(3,3),Q,2.3)
    @test P == Q
    @test x == [2.3,4*2.3,27*2.3]
    x,P = iekfpredict(upowers2,[1.0,2.0,3.0],zeros(3,3),Q,[1,2,3])
    @test P == Q
    @test x == [1.0,8.0,81.0]

    # Analytic tests
    x,P = iekfpredict(powers,powersjac,[1.0,2.0,3.0],zeros(3,3),Q)
    @test P == Q
    @test x == [1.0,4.0,27.0]
    x,P = iekfpredict(upowers1,upowersjac1,[1.0,2.0,3.0],zeros(3,3),Q,2.3)
    @test P == Q
    @test x == [2.3,4*2.3,27*2.3]
    x,P = iekfpredict(upowers2,upowersjac2,[1.0,2.0,3.0],zeros(3,3),Q,[1,2,3])
    @test P == Q
    @test x == [1.0,8.0,81.0]

    # Precomputed Jacobian tests
    x,P = iekfpredict(powers,F1,[1.0,2.0,3.0],zeros(3,3),Q)
    @test P == Q
    @test x == [1.0,4.0,27.0]
    x,P = iekfpredict(upowers1,F2,[1.0,2.0,3.0],zeros(3,3),Q,2.3)
    @test P == Q
    @test x == [2.3,4*2.3,27*2.3]
    x,P = iekfpredict(upowers2,F3,[1.0,2.0,3.0],zeros(3,3),Q,[1,2,3])
    @test P == Q
    @test x == [1.0,8.0,81.0]
end

@testset "predict.jl: P!=0" begin
    Q = rand(3,3)
    Q = Q+Q' # Make Q symmetric
    P = rand(3,3)
    P = P+P' # Make P symmetric

    P1 = F1*P*F1' + Q
    P2 = F2*P*F2' + Q
    P3 = F3*P*F3' + Q

    @test P1 ≈ P1'
    @test P2 ≈ P2'
    @test P3 ≈ P3'

    # Autodiff tests
    x,P_new = iekfpredict(powers,[1.0,2.0,3.0],P,Q)
    @test P_new ≈ P1
    @test x == [1.0,4.0,27.0]
    x,P_new = iekfpredict(upowers1,[1.0,2.0,3.0],P,Q,2.3)
    @test P_new ≈ P2
    @test x == [2.3,4*2.3,27*2.3]
    x,P_new = iekfpredict(upowers2,[1.0,2.0,3.0],P,Q,[1,2,3])
    @test P_new ≈ P3
    @test x == [1.0,8.0,81.0]

    # Analytic tests
    x,P_new = iekfpredict(powers,powersjac,[1.0,2.0,3.0],P,Q)
    @test P_new ≈ P1
    @test x == [1.0,4.0,27.0]
    x,P_new = iekfpredict(upowers1,upowersjac1,[1.0,2.0,3.0],P,Q,2.3)
    @test P_new ≈ P2
    @test x == [2.3,4*2.3,27*2.3]
    x,P_new = iekfpredict(upowers2,upowersjac2,[1.0,2.0,3.0],P,Q,[1,2,3])
    @test P_new ≈ P3
    @test x == [1.0,8.0,81.0]

    # Precomputed Jacobian tests
    x,P_new = iekfpredict(powers,F1,[1.0,2.0,3.0],P,Q)
    @test P_new ≈ P1
    @test x == [1.0,4.0,27.0]
    x,P_new = iekfpredict(upowers1,F2,[1.0,2.0,3.0],P,Q,2.3)
    @test P_new ≈ P2
    @test x == [2.3,4*2.3,27*2.3]
    x,P_new = iekfpredict(upowers2,F3,[1.0,2.0,3.0],P,Q,[1,2,3])
    @test P_new ≈ P3
    @test x == [1.0,8.0,81.0]
end

@testset "predict.jl: integer inputs" begin 
    Q = [1 2 3; 2 3 4; 3 4 5]
    P = [5 4 3; 4 3 2; 3 2 1]

    F1 = powersjac([1,2,3])
    F2 = upowersjac1([1,2,3],2)
    F3 = upowersjac2([1,2,3],[1,2,3])    

    P1 = F1*P*F1' + Q
    P2 = F2*P*F2' + Q
    P3 = F3*P*F3' + Q

    @test P1 ≈ P1'
    @test P2 ≈ P2'
    @test P3 ≈ P3'

    # Autodiff tests
    x,P_new = iekfpredict(powers,[1,2,3],P,Q)
    @test P_new ≈ P1
    @test x == [1,4,27]
    x,P_new = iekfpredict(upowers1,[1,2,3],P,Q,2)
    @test P_new ≈ P2
    @test x == [2,4*2,27*2]
    x,P_new = iekfpredict(upowers2,[1,2,3],P,Q,[1,2,3])
    @test P_new ≈ P3
    @test x == [1,8,81]

    # Analytic tests
    x,P_new = iekfpredict(powers,powersjac,[1,2,3],P,Q)
    @test P_new ≈ P1
    @test x == [1,4,27]
    x,P_new = iekfpredict(upowers1,upowersjac1,[1,2,3],P,Q,2)
    @test P_new ≈ P2
    @test x == [2,4*2,27*2]
    x,P_new = iekfpredict(upowers2,upowersjac2,[1,2,3],P,Q,[1,2,3])
    @test P_new ≈ P3
    @test x == [1,8,81]

    # Precomputed Jacobian tests
    x,P_new = iekfpredict(powers,F1,[1,2,3],P,Q)
    @test P_new ≈ P1
    @test x == [1,4,27]
    x,P_new = iekfpredict(upowers1,F2,[1,2,3],P,Q,2)
end

@testset "predict.jl: scalar system" begin
    x,P_new = iekfpredict(x -> x^2, 2, 1, 1)
    @test x ≈ 4
    @test P_new ≈ 17
    x,P_new = iekfpredict(x -> x^2, x -> 2*x, 2, 1, 1)
    @test x ≈ 4
    @test P_new ≈ 17
end

