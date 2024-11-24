
function h(x)
    1/x
end

function getjac(x)
    -1/x^2
end

function h(x,u)
    u/x
end

function getjac(x,u)
    -u/x^2
end

@testset "correct.jl: P=0" begin
    x,P, residual, Si, Ki, Hi, i = iekfcorrect(h, getjac, 5,1,1/4,2,1,1e-8)
    x2,P2, residual2, Si2, Ki2, Hi2, i2 = iekfcorrect(h, 5,1,1/4,2,1,1e-8)
    # Check that the solution makes sense
    @test x < 5
    @test x > 4
    @test P < 1 # Test that P has decreased
    # Test that the two methods give the same results
    @test x ≈ x2
    @test P ≈ P2
    @test residual ≈ residual2
    @test Si ≈ Si2
    @test Ki ≈ Ki2
    @test Hi ≈ Hi2
    
end