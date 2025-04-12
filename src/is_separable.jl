module is_separable

using Revise
using IJulia
using Ket
using JuMP

import LinearAlgebra
import SCS
import Hypatia
	

k, d = 3, 2
ρ = Ket.random_state(d, k)
X = ρ

function is_ppt(ρ::AbstractMatrix, dims::Vector{Int}, sys::Int=2; atol=1e-10)
    ρ_pt = partial_transpose(ρ, [sys], dims)
    λs = LinearAlgebra.eigen(ρ_pt).values
    return all(λs .>= -atol)
end

function isseparable(X; args...)
    X = Matrix(X)
    if ! LinearAlgebra.isposdef(X)
        error("X is not positive semidefinite, so the idea of it being separable does not make sense.")
    end

    lX = LinearAlgebra.length(X)
    rX = LinearAlgebra.rank(X)
    X = X / LinearAlgebra.tr(X)
    sep = -1

    default_args = (round(sqrt(lX)), 2, 1, eps() ^ (3/8))
    dim, str, verbose, tol = ntuple(i -> i <= length(args) ? args[i] : default_args[i], length(default_args))

    if length(dim) == 1
        dim = [dim, lX / dim]
        if abs(dim[2] - round(dim[2])) >= 2 * lX * eps()
            error("IsSeparable:InvalidDim','If DIM is a scalar, it must evenly divide length(X); please provide the DIM array containing the dimensions of the subsystems.");
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
        return
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
        return
    
    elseif pD <= 6 || minimum(dim) <= 1
        sep = 1
        if Bool(verbose)
            println("Determined to be separable via sufficiency of the PPT criterion in small dimensions. Reference:\n", refs[2], "\n")
        end
        return
    
    elseif rX <= 3 || rX <= LinearAlgebra.rank(XB) || rX <= LinearAlgebra.rank(XA)
        sep = 1
        if Bool(verbose)
            ("Determined to be separable via sufficiency of the PPT criterion for low-rank operators. Reference:\n", refs[3], "\n")
        end
        return
    end
end


main() = isseparable(X)

export main
end # module is_separable
