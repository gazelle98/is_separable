module is_separable

using Revise
using IJulia
using Ket
using JuMP

import LinearAlgebra
import SCS
import Hypatia

k, d = 3, 4
ρ = Ket.random_state(d, k)
X = ρ

function is_ppt(ρ::AbstractMatrix, dims::Vector{Int}, sys::Int=2; atol=1e-10)
    ρ_pt = partial_transpose(ρ, [sys], dims)
    λs = LinearAlgebra.eigen(ρ_pt).values
    return all(λs .>= -atol)
end

function realignment(X::AbstractArray, dim=nothing)
    dX = size(X)
    inferred_dim = round.(sqrt.(dX))

    if dim === nothing
        dim = collect(inferred_dim)
    end

    if length(dim) == 1
        dim = [dim, dX[1] / dim]
        if abs(dim[2] - round(dim[2])) >= 2 * dX[1] * eps()
            error("Realignment:InvalidDim — ",
                "If `dim` is a scalar, `X` must be square and `dim` must evenly divide length(X); ",
                "please provide a two-element `dim` array specifying the subsystem dimensions.")
        end
        dim[2] = round(dim[2])
    end

    if minimum(size(dim)) == 1
        dim = vec(dim)'
        dim = vcat(dim, dim)
    end

    dA, dB = Int.(vec(dim))

    X_tensor = reshape(X, dA, dB, dA, dB)
    X_swapped = permutedims(X_tensor, (2, 1, 4, 3))
    X_swapped_matrix = reshape(X_swapped, dA * dB, dA * dB)

    X_pt = Ket.partial_transpose(X_swapped_matrix, [1], [dB, dA])
    X_pt_tensor = reshape(X_pt, dB, dA, dB, dA)
    X_unswapped = permutedims(X_pt_tensor, (2, 1, 4, 3))

    return reshape(X_unswapped, dA^2, dB^2)
end


function swap_subsystems(X::Matrix{<:Number}, sys::Tuple{Int,Int}, dim::AbstractVector{<:Int})
    d1, d2 = Int.(vec(dim))
    total_dim = d1 * d2
    @assert size(X, 1) == total_dim && size(X, 2) == total_dim "Matrix size doesn't match subsystem dimensions"

    X_tensor = reshape(X, d1, d2, d1, d2)
    X_swapped = permutedims(X_tensor, (2, 1, 4, 3))
    return reshape(X_swapped, total_dim, total_dim)
end


function isseparable(X; args...)
    X = Matrix(X)
    if !LinearAlgebra.isposdef(X)
        error("X is not positive semidefinite, so the idea of it being separable does not make sense.")
    end

    lX = maximum(size(X))
    rX = LinearAlgebra.rank(X)
    X = X / LinearAlgebra.tr(X)
    sep = -1

    default_args = (round(sqrt(lX)), 2, 1, eps()^(3 / 8))
    dim, str, verbose, tol = ntuple(i -> i <= length(args) ? args[i] : default_args[i], length(default_args))

    if length(dim) == 1
        dim = [dim, lX / dim]
        if abs(dim[2] - round(dim[2])) >= 2 * lX * eps()
            error("IsSeparable:InvalidDim','If DIM is a scalar, it must evenly divide length(X); please provide the DIM array containing the dimensions of the subsystems.")
        end
        dim[2] = round(dim[2])
    end

    nD = minimum(dim)
    xD = maximum(dim)
    pD = prod(dim)

    if nD == 1
        sep = 1
        if Bool(verbose)
            println("Every positive semidefinite matrix is separable when one of the local dimensions is 1.")
        end
        return sep
    end

    XA = Ket.partial_trace(X, [2], Int.(dim))
    XB = Ket.partial_trace(X, [1], Int.(dim))

    refs = [
        "A. Peres. Separability criterion for density matrices. Phys. Rev. Lett., 77:1413–1415, 1996.",
        "M. Horodecki, P. Horodecki, and R. Horodecki. Separability of mixed states: Necessary and sufficient conditions. Phys. Lett. A, 223:1–8, 1996.",
        "P. Horodecki, M. Lewenstein, G. Vidal, and I. Cirac. Operational criterion and constructive checks for the separability of low-rank density matrices. Phys. Rev. A, 62:032310, 2000.",
        "K. Chen and L.-A. Wu. A matrix realignment method for recognizing entanglement. Quantum Inf. Comput., 3:193–202, 2003.",
        "F. Verstraete, J. Dehaene, and B. De Moor. Normal forms and entanglement measures for multipartite quantum states. Phys. Rev. A, 68:012103, 2003.",
        "K.-C. Ha and S.-H. Kye. Entanglement witnesses arising from exposed positive linear maps. Open Systems & Information Dynamics, 18:323–337, 2011.",
        "O. Gittsovich, O. Guehne, P. Hyllus, and J. Eisert. Unifying several separability conditions using the covariance matrix criterion. Phys. Rev. A, 78:052319, 2008.",
        "L. Gurvits and H. Barnum. Largest separable balls around the maximally mixed bipartite quantum state. Phys. Rev. A, 66:062311, 2002.",
        "H.-P. Breuer. Optimal entanglement criterion for mixed quantum states. Phys. Rev. Lett., 97:080501, 2006.",
        "W. Hall. Constructions of indecomposable positive maps based on a new criterion for indecomposability. E-print: arXiv:quant-ph/0607035, 2006.",
        "A. C. Doherty, P. A. Parrilo, and F. M. Spedalieri. A complete family of separability criteria. Phys. Rev. A, 69:022308, 2004.",
        "M. Navascues, M. Owari, and M. B. Plenio. A complete criterion for separability detection. Phys. Rev. Lett., 103:160404, 2009.",
        "N. Johnston. Separability from spectrum for qubit-qudit states. Phys. Rev. A, 88:062330, 2013.",
        "C.-J. Zhang, Y.-S. Zhang, S. Zhang, and G.-C. Guo. Entanglement detection beyond the cross-norm or realignment criterion. Phys. Rev. A, 77:060301(R), 2008.",
        "R. Hildebrand. Semidefinite descriptions of low-dimensional separable matrix cones. Linear Algebra Appl., 429:901–932, 2008.",
        "R. Hildebrand. Comparison of the PPT cone and the separable cone for 2-by-n systems. http://www-ljk.imag.fr/membres/Roland.Hildebrand/coreMPseminar2005_slides.pdf",
        "D. Cariello. Separability for weak irreducible matrices. E-print: arXiv:1311.7275 [quant-ph], 2013.",
        "L. Chen and D. Z. Djokovic. Separability problem for multipartite states of rank at most four. J. Phys. A: Math. Theor., 46:275304, 2013.",
        "G. Vidal and R. Tarrach. Robustness of entanglement. Phys. Rev. A, 59:141–155, 1999."
    ]
    
    ppt = is_ppt(X, Int.(dim))

    if !ppt
        sep = 0
        if Bool(verbose)
            println("Determined to be entangled via the PPT criterion. Reference:\n", refs[1], "\n")
        end
        return sep
    
    elseif pD <= 6 || minimum(dim) <= 1
        sep = 1
        if Bool(verbose)
            println("Determined to be separable via sufficiency of the PPT criterion in small dimensions. Reference:\n", refs[2], "\n")
        end
        return sep

    elseif rX <= 3 || rX <= LinearAlgebra.rank(XB) || rX <= LinearAlgebra.rank(XA)
        sep = 1
        if Bool(verbose)
            println("Determined to be separable via sufficiency of the PPT criterion for low-rank operators. Reference:\n", refs[3], "\n")
        end
        return sep
    end

    if Bool(verbose) && Ket.trace_norm(realignment(X, dim)) > 1
        sep = 0
        println("Determined to be entangled via the realignment criterion. Reference:\n$(refs[4])\n")
        return sep
    end

    if Ket.trace_norm(realignment(X - kron(XA, XB))) > sqrt(real((1 - LinearAlgebra.tr(XA^2)) * (1 - LinearAlgebra.tr(XB^2))))
        sep = 0
        if Bool(verbose)
            println("Determined to be entangled by using Theorem 1 of reference:\n", refs[14], "\n")
        end
        return sep
    end

    lam = sort(real.(LinearAlgebra.eigen(X).values), rev=true)

    if nD == 2
        if ((lam[1] - lam[Int(2 * xD - 1)])^2) <= (4 * lam[Int(2 * xD - 2)] * lam[Int(2 * xD)] + tol^2)
            sep = 1
            if Bool(verbose)
                println("Determined to be separable by inspecting its eigenvalues. Reference:\n", refs[13], "\n")
            end
            return sep
        end
        if dim[1] > 2
            Xt = swap_subsystems(X, (1, 2), dim)
        else
            Xt = X
        end
    
        A = Xt[1:Int(xD), 1:Int(xD)]
        B = Xt[1:Int(xD), Int(xD)+1:2*Int(xD)]
        C = Xt[Int(xD)+1:2*Int(xD), Int(xD)+1:2*Int(xD)]
    
        if Bool(verbose) && (LinearAlgebra.norm(B - B') < tol ^ 2)
            sep = 1
            println("Determined to be separable by being a block Hankel matrix:\n", refs[15], "\n")
            return sep
        end
    
        if (LinearAlgebra.rank(B - B') <= 1) && ppt
            sep = 1
            if Bool(verbose)
                println("Determined to be separable by being a perturbed block Hankel matrix:\n", refs[16], "\n")
            end
            return sep
        end
    
        X_2n_ppt_check = vcat(
            hcat((5/6) * A - C / 6, B),
            hcat(B', (5/6) * C - A / 6)
        )
    
        if LinearAlgebra.isposdef(X_2n_ppt_check) && is_ppt(X_2n_ppt_check, [2, Int(xD)])
            sep = 1
            if Bool(verbose)
                println("Determined to be separable via the homothetic images approach of:\n", refs[16], "\n")
            end
            return sep
        end
    end

    println("No separability criterion was satisfied. The state is likely entangled.")
end


main() = begin
    println("Testing is_separable.jl")
    isseparable(X, dim=[4,4], verbose=1)
end

export main
end # module is_separable
