module NumpyArrayInterface

import ArrayInterface

export NpyDescrField, NpyDescr, npy_descr, npy_pointer, npy_readonly, npy_shape, npy_strides, npy_info, npy_mask, npy_array

struct NpyDescrField
    name::String
    descr::Union{String,Vector{NpyDescrField}}
    basic_name::String
    shape::Union{Nothing,Vector{Int}}
end
NpyDescrField(name, descr; basic_name=name, shape=nothing) = NpyDescrField(name, descr, basic_name, shape)

function Base.show(io::IO, d::NpyDescrField)
    show(io, typeof(d))
    print(io, "(")
    show(io, d.name)
    print(io, ", ")
    show(io, d.descr)
    if d.basic_name != d.name
        print(io, ", basic_name=")
        show(io, d.basic_name)
    end
    if d.shape !== nothing
        print(io, ", shape=")
        show(io, d.shape)
    end
    print(io, ")")
end

struct NpyDescr
    typestr::String
    descr::Union{Nothing,Vector{NpyDescrField}}
end
NpyDescr(typestr) = NpyDescr(typestr, nothing)

function Base.show(io::IO, d::NpyDescr)
    show(io, typeof(d))
    print(io, "(")
    show(io, d.typestr)
    if d.descr !== nothing
        print(io, ", ")
        show(io, d.descr)
    end
    print(io, ")")
end

const ENDIAN_CHAR = ENDIAN_BOM == 0x04030201 ? '<' : '>'

const TYPE_TO_DESCR = Dict(
    Bool => NpyDescr("$(ENDIAN_CHAR)b$(sizeof(Bool))"),
    Int8 => NpyDescr("$(ENDIAN_CHAR)i1"),
    Int16 => NpyDescr("$(ENDIAN_CHAR)i2"),
    Int32 => NpyDescr("$(ENDIAN_CHAR)i4"),
    Int64 => NpyDescr("$(ENDIAN_CHAR)i8"),
    UInt8 => NpyDescr("$(ENDIAN_CHAR)u1"),
    UInt16 => NpyDescr("$(ENDIAN_CHAR)u2"),
    UInt32 => NpyDescr("$(ENDIAN_CHAR)u4"),
    UInt64 => NpyDescr("$(ENDIAN_CHAR)u8"),
    Float16 => NpyDescr("$(ENDIAN_CHAR)f2"),
    Float32 => NpyDescr("$(ENDIAN_CHAR)f4"),
    Float64 => NpyDescr("$(ENDIAN_CHAR)f8"),
    Complex{Float16} => NpyDescr("$(ENDIAN_CHAR)c4"),
    Complex{Float32} => NpyDescr("$(ENDIAN_CHAR)c8"),
    Complex{Float64} => NpyDescr("$(ENDIAN_CHAR)c16"),
)

for (T,d) in TYPE_TO_DESCR
    @eval npy_descr(::Type{$T}) = $d
end
npy_descr(T::DataType) = get!(TYPE_TO_DESCR, T) do
    if isstructtype(T)
        n = fieldcount(T)
        if n == 0
            NpyDescr("|V$(sizeof(T))")
        else
            flds = NpyDescrField[]
            for i = 1:n
                nm = fieldname(T, i)
                tp = fieldtype(T, i)
                d = npy_descr(tp)
                name = nm isa Integer ? "f$(nm-1)" : string(nm)
                descr = d.descr === nothing ? d.typestr : d.descr
                push!(flds, NpyDescrField(name, descr))
                off0 = fieldoffset(T, i) + sizeof(tp)
                off1 = i == n ? sizeof(T) : fieldoffset(T, i + 1)
                d = off1 - off0
                @assert d ≥ 0
                d > 0 && push!(flds, NpyDescrField("", "|V$(d)"))
            end
            NpyDescr("|V$(sizeof(T))", flds)
        end
    elseif isbitstype(T)
        NpyDescr("|V$(sizeof(T))")
    else
        @assert false
    end
end

npy_array(x) = nothing
npy_array(x::StridedArray) = x  # TODO

npy_descr(x) = npy_descr(eltype(x))

npy_shape(x) = Tuple{Vararg{Int}}(size(x))

npy_strides(x) = Tuple{Vararg{Int}}(strides(x) .* Base.aligned_sizeof(eltype(x)))

npy_pointer(x) = Ptr{Cvoid}(Base.unsafe_convert(Ptr{eltype(x)}, x))

npy_readonly(x) = !ArrayInterface.can_setindex(typeof(x))

npy_mask(x) = nothing

npy_info(x) = if (a = npy_array(x)) !== nothing
    (
        array = a,
        pointer = npy_pointer(a),
        readonly = npy_readonly(a),
        shape = npy_shape(a),
        descr = npy_descr(a),
        strides = npy_strides(a),
        mask = npy_mask(a),
    )
end

end # module
