struct ConvOp{T} <: AbstractConvOp
    data::Array{T,3}
    k0::Array{Int,2}
    k1::Array{Int,2}
    tail::Array{T,2}
    length::Int
end

function Base.eltype(x::ConvOp)
    return eltype(x.data)
end

function Base.size(obj::ConvOp) return (size(obj.data)[2:3]...,obj.length) end
function _getindex(obj::ConvOp, m::Int, n::Int, k::Int)
    if k < obj.k0[m,n]
        return zero(eltype(obj.data))
    end

    if k > obj.k1[m,n]
        return obj.tail[m,n]
    end

    return obj.data[k-obj.k0[m,n]+1,m,n]
end



function convolve!(y, Z::ConvOp, x, X, j, k_start=1, k_stop=size(Z,3))
    @assert k_stop >= k_start
    # Each task owns a contiguous block of rows (no write races). Within a block
    # we loop n outer / m inner so that data[:, mrange, n], k0/k1/tail[mrange, n]
    # are traversed contiguously in memory and x[n, :] is reused across rows.
    M = length(axes(y,1))
    M == 0 && return y
    nchunks = min(M, 4 * Threads.nthreads())
    @sync for mrange in Iterators.partition(axes(y,1), cld(M, nchunks))
        Threads.@spawn _convolve_rows!(y, Z, x, X, j, k_start, k_stop, mrange)
    end
    return y
end

function _convolve_rows!(y, Z::ConvOp, x, X, j, k_start, k_stop, mrange)
    T = eltype(y)
    @inbounds for n in axes(x,1)
        for m in mrange
            k0 = Z.k0[m,n]
            k1 = Z.k1[m,n]

            # banded part; the bound k <= j replaces the `j-k < 0` test
            s = zero(T)
            @simd for k in max(k0,k_start):min(k1,k_stop,j)
                s += Z.data[k-k0+1,m,n] * x[n,j-k+1]
            end

            # constant tail part, evaluated via the cumulative sum X
            q0 = j - min(k_stop, j) + 1
            q1 = j - max(k1, k_start-1)
            if q1 >= q0
                s += q0 > 1 ? Z.tail[m,n] * (X[n,q1] - X[n,q0-1]) :
                              Z.tail[m,n] * X[n,q1]
            end
            y[m] += s
        end
    end
    return y
end

function timeslice!(Y, Z::ConvOp, k)
    @assert size(Y) == size(Z)[1:2]
    for m in axes(Z,1)
        for n in axes(Z,2)
            Y[m,n] += _getindex(Z, m,n,k)
    end end
    return Y 
end

struct ConvOpAsArray{T} <: AbstractArray{T,3}
    convop::ConvOp{T}
end

function Base.size(x::ConvOpAsArray) size(x.convop) end
function Base.getindex(x::ConvOpAsArray, I::Vararg{Int,3}) _getindex(x.convop,I...) end


struct ConvOpAsMatrix{T} <: AbstractMatrix{Vector{T}}
    convop::ConvOp{T}
end

function Base.size(x::ConvOpAsMatrix) size(x.convop)[1:2] end
function Base.getindex(x::ConvOpAsMatrix, I::Vararg{Int,2})
    [_getindex(x.convop,I...,k) for k in axes(x.convop,3)]
end

function Base.AbstractArray(x::ConvOp{T}) where {T} ConvOpAsArray(x) end

function convolve!(y, Z::ConvOpAsMatrix, x, X, j, k_start=1, k_stop=size(Z,3))
    convolve!(y, Z.convop, x, X, j, k_start, k_stop)
end

function tailindex(Z::ConvOp)
    return maximum(Z.k1)
end

"""
    hastail(x::AbstractConvop) -> Bool

Returns yes if the timeslices for k larger then `tailindex(x)` take on
    a constant value. This can be exploited to compute convolutions more
    efficiency.
"""
function hastail(Z::ConvOp)
    return Z.length > tailindex(Z)
end