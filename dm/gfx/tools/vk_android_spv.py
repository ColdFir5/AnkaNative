"""SPIR-V reading and the few rewrites the Android Vulkan library needs (vk_android_build.py): the descriptor set / binding renumbering into the Vulkan ABI
(dm/gfx/nvn/dmsh_abi_vk.h), vertex outputs that start as (0, 0, 0, 1), the GL depth and alpha test constants, the integer-output and dual-source variants, the
viewport mask of an emulated passthrough geometry shader, and the facts the reflection reads (resources, vertex inputs, colour outputs, capabilities). The translator's modules are taken as they come; every rewrite keeps the module valid for spirv-val --target-env vulkan1.3."""
import array, itertools, struct, sys

MAGIC = 0x07230203
_WORDS_LE = array.array('I').itemsize == 4 and sys.byteorder == 'little'   # (Module.data / parse through array: the words as they are in memory)
# opcodes
OP_NAME, OP_EXTENSION, OP_EXT_INST_IMPORT, OP_EXT_INST, OP_ENTRY_POINT, OP_EXECUTION_MODE, OP_CAPABILITY = 5, 10, 11, 12, 15, 16, 17
OP_TYPE_VOID, OP_TYPE_BOOL, OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR, OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE = 19, 20, 21, 22, 23, 25, 26, 27
OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER = 28, 29, 30, 32
OP_CONSTANT, OP_CONSTANT_COMPOSITE, OP_FUNCTION, OP_FUNCTION_END, OP_VARIABLE, OP_LOAD, OP_STORE = 43, 44, 54, 56, 59, 61, 62
OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN, OP_DECORATE, OP_MEMBER_DECORATE = 65, 66, 71, 72
OP_COMPOSITE_EXTRACT, OP_COMPOSITE_INSERT, OP_COPY_OBJECT, OP_BITCAST, OP_FADD, OP_FMUL = 81, 82, 83, 124, 129, 133
OP_EMIT_VERTEX, OP_EMIT_STREAM_VERTEX, OP_LABEL, OP_RETURN, OP_RETURN_VALUE, OP_IMAGE_TEXEL_POINTER = 218, 220, 248, 253, 254, 60
OP_SPEC_CONSTANT_FALSE, OP_SELECT, OP_IMAGE_READ, OP_IMAGE_WRITE = 49, 169, 98, 99
# decorations, storage classes, builtins
DEC_SPEC_ID, DEC_BLOCK, DEC_BUFFER_BLOCK, DEC_BUILTIN, DEC_LOCATION, DEC_BINDING, DEC_SET = 1, 2, 3, 11, 30, 33, 34
SC_UNIFORM_CONSTANT, SC_INPUT, SC_UNIFORM, SC_OUTPUT, SC_FUNCTION, SC_PUSH_CONSTANT, SC_STORAGE_BUFFER = 0, 1, 2, 3, 7, 9, 12
BUILTIN_POSITION = 0
# VkDescriptorType
VK_SAMPLER, VK_COMBINED, VK_SAMPLED_IMAGE, VK_STORAGE_IMAGE, VK_UNIFORM_TEXEL, VK_STORAGE_TEXEL, VK_UNIFORM_BUFFER, VK_STORAGE_BUFFER = 0, 1, 2, 3, 4, 5, 6, 7
DIM_BUFFER = 5
EXEC_MODELS = {0: 'vs', 1: 'hs', 2: 'ds', 3: 'gs', 4: 'ps', 5: 'cs', 5364: 'ts', 5365: 'ms'}
CAPS = {1: 'Shader', 2: 'Geometry', 3: 'Tessellation', 25: 'ImageGatherExtended', 27: 'StorageImageMultisample', 34: 'ImageCubeArray', 35: 'SampleRateShading',
        43: 'Sampled1D', 44: 'Image1D', 45: 'SampledCubeArray', 46: 'SampledBuffer', 47: 'ImageBuffer', 48: 'ImageMSArray', 49: 'StorageImageExtendedFormats', 50: 'ImageQuery',
        51: 'DerivativeControl', 52: 'InterpolationFunction', 54: 'GeometryStreams', 4427: 'DrawParameters', 9: 'Float16', 10: 'Float64', 11: 'Int64', 12: 'Int64Atomics', 22: 'Int16', 39: 'Int8', 32: 'ClipDistance', 33: 'CullDistance', 53: 'TransformFeedback',
        55: 'StorageImageReadWithoutFormat', 56: 'StorageImageWriteWithoutFormat', 57: 'MultiViewport', 61: 'GroupNonUniform', 62: 'GroupNonUniformVote',
        63: 'GroupNonUniformArithmetic', 64: 'GroupNonUniformBallot', 65: 'GroupNonUniformShuffle', 66: 'GroupNonUniformShuffleRelative', 67: 'GroupNonUniformClustered',
        68: 'GroupNonUniformQuad', 4423: 'SubgroupBallotKHR', 4431: 'SubgroupVoteKHR', 5251: 'GeometryShaderPassthroughNV', 5254: 'ShaderViewportIndexLayerEXT',
        5255: 'ShaderViewportMaskNV', 5347: 'PhysicalStorageBufferAddresses', 5363: 'FragmentShaderSampleInterlockEXT', 5372: 'FragmentShaderShadingRateInterlockEXT',
        5378: 'FragmentShaderPixelInterlockEXT', 5379: 'DemoteToHelperInvocation', 6022: 'CooperativeMatrixKHR', 6033: 'AtomicFloat32AddEXT',
        5013: 'ImageGatherBiasLodAMD', 5568: 'IntegerFunctions2INTEL', 5283: 'MeshShadingEXT'}


def string_at(words, i):
    """A nul-terminated literal string starting at words[i] → (text, index of the word after it)"""
    out = bytearray()
    while i < len(words):
        b = struct.pack('<I', words[i]); i += 1
        if 0 in b: out += b[:b.index(0)]; break
        out += b
    return out.decode('utf-8', 'replace'), i


def lit_string(text):
    b = text.encode() + b'\0'
    b += b'\0' * (-len(b) % 4)
    return list(struct.unpack('<%dI' % (len(b) // 4), b))


class Module:
    def __init__(self, data):
        if len(data) % 4 or len(data) < 20: raise ValueError('not a SPIR-V module (length)')
        w = array.array('I', data).tolist() if _WORDS_LE else list(struct.unpack('<%dI' % (len(data) // 4), data))
        if w[0] != MAGIC: raise ValueError('not a SPIR-V module (magic)')
        self.header, self.ins, i, end = w[:5], [], 5, len(w)
        add = self.ins.append   # (the build parses every module several times: local names)
        while i < end:
            n = w[i] >> 16
            if not n or i + n > end: raise ValueError('malformed SPIR-V instruction at word %d' % i)
            add(w[i:i + n]); i += n
        self.index()

    def index(self):
        self.names, self.decos, self.types, self.consts, self.vars, self.caps, self.exts = {}, {}, {}, {}, {}, [], []
        self.entry = None; self.modes = []
        for x in self.ins:
            op = x[0] & 0xffff
            if op == OP_NAME: self.names[x[1]] = string_at(x, 2)[0]
            elif op == OP_DECORATE: self.decos.setdefault(x[1], {})[x[2]] = x[3:]
            elif op == OP_CAPABILITY: self.caps.append(x[1])
            elif op == OP_EXTENSION: self.exts.append(string_at(x, 1)[0])
            elif op == OP_ENTRY_POINT: self.entry = x
            elif op == OP_EXECUTION_MODE: self.modes.append(x[2:])
            elif 19 <= op <= 39: self.types[x[1]] = x
            elif op == OP_CONSTANT: self.consts[x[2]] = (x[1], x[3:])
            elif op == OP_VARIABLE and len(x) >= 4: self.vars[x[2]] = (x[1], x[3])

    def data(self):
        if _WORDS_LE:   # (the same bytes: one list of every word, made in C, then an array's copy instead of one argument per word)
            return array.array('I', self.header + list(itertools.chain.from_iterable(self.ins))).tobytes()
        return struct.pack('<%dI' % (5 + sum(len(x) for x in self.ins)), *self.header, *(v for x in self.ins for v in x))

    def new_id(self):
        i = self.header[3]; self.header[3] += 1; return i

    @property
    def model(self): return EXEC_MODELS.get(self.entry[1]) if self.entry else None

    def interface(self):
        _, k = string_at(self.entry, 3)
        return self.entry[k:]

    def op(self, ident): return self.types[ident][0] & 0xffff if ident in self.types else None

    def pointee(self, ptr):   # OpTypePointer → (storage class, type)
        t = self.types[ptr]; return t[2], t[3]

    def strip_arrays(self, t):
        """(element type, descriptor count): count 0 for a runtime array"""
        count = 1
        while self.op(t) in (OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY):
            x = self.types[t]
            if self.op(t) == OP_TYPE_RUNTIME_ARRAY: count = 0; t = x[2]; continue
            n = self.consts.get(x[3], (0, [0]))[1][0]
            count *= n; t = x[2]
        return t, count

    def int_type(self, t):   # (width, signed) of an OpTypeInt, or None
        x = self.types.get(t)
        return (x[2], x[3]) if x and x[0] & 0xffff == OP_TYPE_INT else None

    def scalar_of(self, t):   # the component type of a scalar or vector
        return self.types[t][2] if self.op(t) == OP_TYPE_VECTOR else t

    def find_type(self, op, *operands):
        for i, x in self.types.items():
            if x[0] & 0xffff == op and x[2:] == list(operands): return i
        return None

    def add_type(self, op, *operands):
        """An existing type with these operands, or a new one placed after the last type / constant declaration"""
        i = self.find_type(op, *operands)
        if i is not None: return i
        i = self.new_id(); x = [(2 + len(operands)) << 16 | op, i, *operands]
        self._insert_decl(x); self.types[i] = x
        return i

    def add_const(self, ty, value):
        for i, (t, v) in self.consts.items():
            if t == ty and v == [value]: return i
        i = self.new_id(); x = [4 << 16 | OP_CONSTANT, ty, i, value]
        self._insert_decl(x); self.consts[i] = (ty, [value])
        return i

    def _insert_decl(self, x):   # after the last type, constant or global variable: before the first function
        at = next(k for k, y in enumerate(self.ins) if y[0] & 0xffff == OP_FUNCTION)
        self.ins.insert(at, x)

    def resources(self):
        """Descriptor variables → [{id, name, set, binding, desc, count, dim, rw}] in declaration order"""
        out = []
        for vid, (ptr, sc) in self.vars.items():
            d = self.decos.get(vid, {})
            if DEC_BINDING not in d: continue
            _, t = self.pointee(ptr)
            elem, count = self.strip_arrays(t)
            e = {'id': vid, 'name': self.names.get(vid, ''), 'set': d.get(DEC_SET, [0])[0], 'binding': d[DEC_BINDING][0], 'count': count, 'dim': None}
            eop = self.op(elem)
            if sc == SC_UNIFORM and self.op(elem) == OP_TYPE_STRUCT:
                e['desc'] = VK_STORAGE_BUFFER if DEC_BUFFER_BLOCK in self.decos.get(elem, {}) else VK_UNIFORM_BUFFER
            elif sc == SC_STORAGE_BUFFER: e['desc'] = VK_STORAGE_BUFFER
            elif sc == SC_UNIFORM_CONSTANT:
                if eop == OP_TYPE_SAMPLER: e['desc'] = VK_SAMPLER
                else:
                    img = self.types[elem][2] if eop == OP_TYPE_SAMPLED_IMAGE else elem
                    if self.op(img) != OP_TYPE_IMAGE: raise ValueError(f'resource {e["name"]}: type {eop} is not an image, sampler or block')
                    _, _, sampled_t, dim, depth, arrayed, ms, sampled = self.types[img][:8]
                    e['dim'] = {'dim': dim, 'depth': depth == 1, 'array': bool(arrayed), 'ms': bool(ms), 'type': self.int_type(sampled_t)}
                    if eop == OP_TYPE_SAMPLED_IMAGE: e['desc'] = VK_COMBINED
                    elif sampled == 2: e['desc'] = VK_STORAGE_TEXEL if dim == DIM_BUFFER else VK_STORAGE_IMAGE
                    else: e['desc'] = VK_UNIFORM_TEXEL if dim == DIM_BUFFER else VK_SAMPLED_IMAGE
            else: raise ValueError(f'resource {e["name"]}: storage class {sc}')
            out.append(e)
        return out

    def set_binding(self, vid, s, b):
        for x in self.ins:
            if x[0] & 0xffff == OP_DECORATE and x[1] == vid:
                if x[2] == DEC_SET: x[3] = s
                elif x[2] == DEC_BINDING: x[3] = b
        d = self.decos.setdefault(vid, {})
        if DEC_SET not in d:   # a resource without a set decoration is in set 0: give it its set explicitly
            at = next(k for k, y in enumerate(self.ins) if y[0] & 0xffff == OP_DECORATE and y[1] == vid)
            self.ins.insert(at, [4 << 16 | OP_DECORATE, vid, DEC_SET, s])
        d[DEC_SET] = [s]; d[DEC_BINDING] = [b]

    def io(self, sc):
        """Location-decorated Input / Output variables of the entry point → {location: (variable, value type)}"""
        out = {}
        for vid in self.interface():
            ptr, c = self.vars.get(vid, (None, None))
            if c != sc: continue
            d = self.decos.get(vid, {})
            if DEC_LOCATION in d: out[d[DEC_LOCATION][0]] = (vid, self.pointee(ptr)[1])
        return out

    def builtin_var(self, sc, builtin):
        """(variable, member index or None) of a builtin in the interface: a variable decorated with it, or a block member"""
        for vid in self.interface():
            ptr, c = self.vars.get(vid, (None, None))
            if c != sc: continue
            if self.decos.get(vid, {}).get(DEC_BUILTIN) == [builtin]: return vid, None
            t = self.pointee(ptr)[1]
            for x in self.ins:
                if x[0] & 0xffff == OP_MEMBER_DECORATE and x[1] == t and x[3] == DEC_BUILTIN and x[4] == builtin: return vid, x[2]
        return None, None

    def entry_function(self): return self.entry[2]

    def function_range(self, fid):   # (index of OpFunction, index of OpFunctionEnd)
        a = next(k for k, x in enumerate(self.ins) if x[0] & 0xffff == OP_FUNCTION and x[2] == fid)
        return a, next(k for k in range(a, len(self.ins)) if self.ins[k][0] & 0xffff == OP_FUNCTION_END)

    def uses(self):
        """id → [instruction indices that take it as an operand] (operand positions are not decoded: a literal equal to an id counts too, which only widens the search)"""
        u = {}
        for k, x in enumerate(self.ins):
            for v in x[1:]: u.setdefault(v, []).append(k)
        return u


def defined_id(x):
    """The result id of an instruction with a result type (or a type / label), else None: [opcode, type, result, ...] for most"""
    op = x[0] & 0xffff
    if 19 <= op <= 39 or op in (OP_LABEL, OP_EXT_INST_IMPORT): return x[1]
    if op in (OP_LOAD, OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN, OP_COMPOSITE_EXTRACT, OP_COPY_OBJECT, OP_BITCAST, OP_VARIABLE, OP_CONSTANT, OP_CONSTANT_COMPOSITE): return x[2]
    return None


def first_block_insert_point(m, fid):
    """The index in m.ins right after the entry function's first label and its OpVariable declarations"""
    a, _ = m.function_range(fid)
    k = next(k for k in range(a, len(m.ins)) if m.ins[k][0] & 0xffff == OP_LABEL) + 1
    while m.ins[k][0] & 0xffff == OP_VARIABLE: k += 1
    return k


def vec4_const(m, f32, vals):
    vt = m.add_type(OP_TYPE_VECTOR, f32, 4)
    ids = [m.add_const(f32, struct.unpack('<I', struct.pack('<f', v))[0]) for v in vals]
    for i, x in enumerate(m.ins):
        if x[0] & 0xffff == OP_CONSTANT_COMPOSITE and x[1] == vt and x[3:] == ids: return x[2]
    c = m.new_id(); m._insert_decl([7 << 16 | OP_CONSTANT_COMPOSITE, vt, c, *ids])
    return c


def init_outputs(m, extra_locations=()):
    """Vertex outputs start as (0, 0, 0, 1) before the body runs, as an unwritten component reads on the hardware (dmsh_build.py vs_init); extra_locations get new
    vec4 outputs holding that value, for paired pixel shaders that read what this shader does not write (vs_pad). → the locations added"""
    outs = m.io(SC_OUTPUT)
    f32 = m.find_type(OP_TYPE_FLOAT, 32)
    if f32 is None: return []
    one = vec4_const(m, f32, (0.0, 0.0, 0.0, 1.0))
    vt = m.find_type(OP_TYPE_VECTOR, f32, 4)
    added = []
    for loc in sorted(set(extra_locations) - set(outs)):
        ptr = m.add_type(OP_TYPE_POINTER, SC_OUTPUT, vt)
        v = m.new_id(); m._insert_decl([4 << 16 | OP_VARIABLE, ptr, v, SC_OUTPUT]); m.vars[v] = (ptr, SC_OUTPUT)
        at = next(k for k, y in enumerate(m.ins) if y[0] & 0xffff == OP_DECORATE)
        m.ins.insert(at, [4 << 16 | OP_DECORATE, v, DEC_LOCATION, loc]); m.decos[v] = {DEC_LOCATION: [loc]}
        name = lit_string(f'dm_pad{loc}'); m.ins.insert(at, [(2 + len(name)) << 16 | OP_NAME, v, *name])
        ep = m.entry; ep.append(v); ep[0] = len(ep) << 16 | OP_ENTRY_POINT
        outs[loc] = (v, vt); added.append(loc)
    k = first_block_insert_point(m, m.entry_function())
    stores = [[3 << 16 | OP_STORE, v, one] for loc, (v, t) in sorted(outs.items()) if t == vt]
    m.ins[k:k] = stores
    return added


def gl_depth(m, spec_id):
    """The GL depth convention (near = -w, NVN's default) under a specialization constant: z' = fma(z, 0.5, w * 0.5) of the position output, selected when the bool
    constant spec_id is true (default false: z as translated, NVN's near = 0), before every return of the entry function or before every vertex a geometry shader emits
    (dmsh_build.py vs_depth, the same arithmetic; there it is a second entry point). → True when the module writes a position"""
    pos, member = m.builtin_var(SC_OUTPUT, BUILTIN_POSITION)
    if pos is None: return False
    f32 = m.find_type(OP_TYPE_FLOAT, 32); vt = m.find_type(OP_TYPE_VECTOR, f32, 4)
    pvec = m.add_type(OP_TYPE_POINTER, SC_OUTPUT, vt)
    half = m.add_const(f32, 0x3f000000)
    i32 = m.find_type(OP_TYPE_INT, 32, 1) or m.add_type(OP_TYPE_INT, 32, 1)
    midx = m.add_const(i32, member) if member is not None else None
    glsl = ensure_glsl(m)
    boolean = m.add_type(OP_TYPE_BOOL)
    spec = m.new_id(); m._insert_decl([3 << 16 | OP_SPEC_CONSTANT_FALSE, boolean, spec])
    at = next(k for k, y in enumerate(m.ins) if y[0] & 0xffff == OP_DECORATE)
    m.ins.insert(at, [4 << 16 | OP_DECORATE, spec, DEC_SPEC_ID, spec_id]); m.decos[spec] = {DEC_SPEC_ID: [spec_id]}
    name = lit_string('dm_gl_depth'); m.ins.insert(next(k for k, y in enumerate(m.ins) if y[0] & 0xffff in (OP_NAME, OP_DECORATE)), [(2 + len(name)) << 16 | OP_NAME, spec, *name])
    a, e = m.function_range(m.entry_function())
    stop = (OP_EMIT_VERTEX, OP_EMIT_STREAM_VERTEX) if m.model == 'gs' else (OP_RETURN, OP_RETURN_VALUE)
    out, n = [], 0
    for k, x in enumerate(m.ins):
        if a < k < e and x[0] & 0xffff in stop:
            p = pos
            if midx is not None: p = m.new_id(); out.append([5 << 16 | OP_ACCESS_CHAIN, pvec, p, pos, midx])
            v, z, w, wh, z2, z3, v2 = (m.new_id() for _ in range(7))
            out += [[4 << 16 | OP_LOAD, vt, v, p], [5 << 16 | OP_COMPOSITE_EXTRACT, f32, z, v, 2], [5 << 16 | OP_COMPOSITE_EXTRACT, f32, w, v, 3],
                    [5 << 16 | OP_FMUL, f32, wh, w, half], [8 << 16 | OP_EXT_INST, f32, z2, glsl, 50, z, half, wh],   # GLSL.std.450 50: Fma
                    [6 << 16 | OP_SELECT, f32, z3, spec, z2, z], [6 << 16 | OP_COMPOSITE_INSERT, vt, v2, z3, v, 2], [3 << 16 | OP_STORE, p, v2]]
            n += 1
        out.append(x)
    m.ins = out
    return n > 0


def value_uses(m):
    """id → [(instruction, operand position)] for the instructions of function bodies whose operands are ids at known positions"""
    u = {}
    for x in m.ins:
        op = x[0] & 0xffff
        if op in (OP_LOAD,): ids = [(3, x[3])]
        elif op == OP_STORE: ids = [(1, x[1]), (2, x[2])]
        elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN): ids = [(p, x[p]) for p in range(3, len(x))]
        elif op == OP_COMPOSITE_EXTRACT: ids = [(3, x[3])]
        elif op in (OP_COPY_OBJECT, OP_BITCAST): ids = [(3, x[3])]
        elif op == OP_ENTRY_POINT or op == OP_NAME or op == OP_DECORATE or op == OP_MEMBER_DECORATE: continue
        else: ids = [(p, x[p]) for p in range(1, len(x))]   # conservative: every word may be an id
        for p, v in ids: u.setdefault(v, []).append((x, p))
    return u


def integer_only_inputs(m):
    """Vertex inputs declared float that the shader only uses as integer bits (dmsh_build.py int_inputs, here on the SPIR-V): every value loaded from the input, whole or
    by component, is only bit-cast to an integer type, copied, or stored to a function variable whose loads are such values in turn. → {location: 'uint'}"""
    u = value_uses(m)
    def int_bits(val, seen):
        if val in seen: return True
        seen.add(val)
        users = u.get(val, [])
        if not users: return False
        for x, p in users:
            op = x[0] & 0xffff
            if op == OP_BITCAST and m.int_type(m.scalar_of(x[1])): continue
            if op in (OP_COPY_OBJECT, OP_COMPOSITE_EXTRACT) and p == 3:
                if not int_bits(x[2], seen): return False
                continue
            if op == OP_STORE and p == 2:   # into a local: its loads must be integer uses too
                var = x[1]
                if m.vars.get(var, (0, None))[1] != SC_FUNCTION: return False
                loads = [y[2] for y, q in u.get(var, []) if y[0] & 0xffff == OP_LOAD and q == 3]
                if any(y[0] & 0xffff not in (OP_LOAD, OP_STORE) for y, _ in u.get(var, [])): return False
                if not all(int_bits(l, seen) for l in loads): return False
                continue
            return False
        return True
    out = {}
    for loc, (var, t) in m.io(SC_INPUT).items():
        if m.int_type(m.scalar_of(t)): continue
        vals, ok = [], True
        for x, p in u.get(var, []):
            op = x[0] & 0xffff
            if op == OP_LOAD and p == 3: vals.append(x[2])
            elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and p == 3:
                for y, q in u.get(x[2], []):
                    if y[0] & 0xffff == OP_LOAD and q == 3: vals.append(y[2])
                    else: ok = False
            else: ok = False
        if ok and vals and all(int_bits(v, set()) for v in vals): out[str(loc)] = 'uint'
    return out


def int_outputs(m):
    """Colour outputs (fragment Output locations) some store writes from integer bits (a bit cast of an integer value, whole or through an access chain).
    Three passes over the instructions (the outputs' access chains, the stores through them, the bit casts of what they store) instead of value_uses: the same set"""
    ptr = {}   # an output variable or an access chain on one → its locations
    for loc, (var, _) in m.io(SC_OUTPUT).items(): ptr.setdefault(var, set()).add(loc)
    if not ptr: return set()
    var_locs = dict(ptr)
    for x in m.ins:
        if x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and len(x) > 3 and x[3] in var_locs: ptr.setdefault(x[2], set()).update(var_locs[x[3]])
    stored = {}   # a value some store writes through those pointers → the locations
    for x in m.ins:
        if x[0] & 0xffff == OP_STORE and len(x) > 2 and x[1] in ptr: stored.setdefault(x[2], set()).update(ptr[x[1]])
    if not stored: return set()
    defs = {}   # (the last definition of an id, as a scan of every instruction gives it)
    for x in m.ins:
        d = defined_id(x)
        if d is not None and d in stored: defs[d] = x[0] & 0xffff
    return {loc for v, locs in stored.items() if defs.get(v) == OP_BITCAST for loc in locs}   # the translator bit-casts only between integers and floats


def functions_named(m):
    return {m.names.get(x[2], ''): x[2] for x in m.ins if x[0] & 0xffff == OP_FUNCTION}


def ensure_glsl(m):
    """The GLSL.std.450 import's id, added before OpMemoryModel when the module has none"""
    for x in m.ins:
        if x[0] & 0xffff == OP_EXT_INST_IMPORT and string_at(x, 2)[0] == 'GLSL.std.450': return x[1]
    i = m.new_id(); s = lit_string('GLSL.std.450')
    at = next(k for k, x in enumerate(m.ins) if x[0] & 0xffff == 14)   # OpMemoryModel
    m.ins.insert(at, [(2 + len(s)) << 16 | OP_EXT_INST_IMPORT, i, *s])
    return i


SUBGROUP_OPS = set(range(333, 367)) | {4421, 4422, 4428, 4429, 4430, 4432}   # OpGroupNonUniform*, the KHR ballot / vote instructions
SUBGROUP_BUILTINS = {36, 37, 40, 41, 4416, 4417, 4418, 4419, 4420}   # SubgroupSize, NumSubgroups, SubgroupId, SubgroupLocalInvocationId, the lane masks
CAP_SUBGROUP = {61, 62, 63, 64, 65, 66, 67, 68, 4423, 4431}
CAP_FLOAT64, CAP_INTERLOCK = 10, {5363, 5372, 5378}
OP_TYPE_FLOAT64_WIDTH = 64
INTERLOCK_OPS, INTERLOCK_MODES = {5364, 5365}, {5366, 5367, 5368, 5369, 5370, 5371}   # Begin / EndInvocationInterlockEXT; the interlock execution modes


def subgroup_used(m):
    """A subgroup instruction, or a subgroup builtin (a decorated variable or a block member)"""
    return _scan_caps(m)[2]


def _scan_caps(m):
    """One pass over the instructions → (a 64-bit float type, an interlock instruction, subgroup_used)"""
    f64 = inter = sg = False
    for x in m.ins:
        op = x[0] & 0xffff
        if op == OP_TYPE_FLOAT:
            if x[2] == 64: f64 = True
        elif op in SUBGROUP_OPS: sg = True
        elif op == OP_MEMBER_DECORATE:
            if x[3] == DEC_BUILTIN and x[4] in SUBGROUP_BUILTINS: sg = True
        elif op in INTERLOCK_OPS: inter = True
    if not sg: sg = any(d.get(DEC_BUILTIN, [None])[0] in SUBGROUP_BUILTINS for d in m.decos.values())
    return f64, inter, sg


def strip_unused_caps(m):
    """Capabilities the translator declares in every module whatever it uses (its host answers say the device has them), dropped where nothing needs them, so a module asks the device
    only for what it uses: Float64 without a 64-bit float type (the phone has no shaderFloat64; a shader that used doubles was already lowered, Program.cs softfloat64), fragment shader
    interlock without an interlock instruction (with its extension and modes), the subgroup capabilities without a subgroup instruction or builtin. → names of those dropped"""
    drop = set()
    f64, inter, sg = _scan_caps(m)
    if not f64: drop.add(CAP_FLOAT64)
    if not inter: drop |= CAP_INTERLOCK
    if not sg: drop |= CAP_SUBGROUP
    drop &= set(m.caps)
    keep = []
    for x in m.ins:
        op = x[0] & 0xffff
        if op == OP_CAPABILITY and x[1] in drop: continue
        if op == OP_EXTENSION and drop & CAP_INTERLOCK and string_at(x, 1)[0] == 'SPV_EXT_fragment_shader_interlock': continue
        if op == OP_EXECUTION_MODE and drop & CAP_INTERLOCK and x[2] in INTERLOCK_MODES: continue
        keep.append(x)
    m.ins = keep; m.index()
    return sorted(CAPS.get(c, str(c)) for c in drop)


GLSL_PRECISE = {37: 79, 40: 80, 43: 81}   # GLSL.std.450 FMin / FMax / FClamp → NMin / NMax / NClamp


def precise_minmax(m):
    """min, max and clamp with Maxwell's NaN results: FMNMX returns the other operand of a NaN and saturation gives 0, which GLSL.std.450's NMin / NMax / NClamp
    define and FMin / FMax / FClamp leave to the device (dmsh_build.py precise, the same rule for Metal's fast:: functions). → instructions rewritten"""
    glsl = {x[1] for x in m.ins if x[0] & 0xffff == OP_EXT_INST_IMPORT and string_at(x, 2)[0] == 'GLSL.std.450'}
    n = 0
    for x in m.ins:
        if x[0] & 0xffff == OP_EXT_INST and x[3] in glsl and x[4] in GLSL_PRECISE and m.op(x[1]) != OP_TYPE_INT and m.op(m.scalar_of(x[1])) == OP_TYPE_FLOAT:
            x[4] = GLSL_PRECISE[x[4]]; n += 1
    return n


OP_IMAGE_FETCH = 95
IMG_OPERAND_WORDS = ((0x1, 1), (0x2, 1), (0x4, 2), (0x8, 1), (0x10, 1), (0x20, 1), (0x40, 1), (0x80, 1))   # Bias, Lod, Grad (2), ConstOffset, Offset, ConstOffsets, Sample, MinLod
def fetch_offset(m):
    """OpImageFetch with a non-constant Offset (TLD with a register offset) as the coordinate plus the offset: Vulkan takes the Offset image operand only on gathers without
    maintenance8 (VUID-RuntimeSpirv-Offset-10213), so spirv-val refused the module and the shader was not in the library (Ocarina of Time's AA pass d17eefaff4f0bc23,
    10 fetches). A fetch's offset adds to its integer texel coordinate at the level it reads, so the add is exact. Fetches whose coordinate and offset types differ (an arrayed
    coordinate) are left. → fetches rewritten"""
    ty, body = {}, False
    for x in m.ins:   # (result types of the values in functions: outside them _result_id's [op, type, result] shape does not hold)
        op = x[0] & 0xffff
        if op == OP_FUNCTION: body = True
        elif op == OP_FUNCTION_END: body = False
        if not body or op == OP_LABEL_: continue
        r = _result_id(x)
        if r is not None and len(x) > 2 and r == x[2]: ty[r] = x[1]
    for i, (t, _) in m.consts.items(): ty.setdefault(i, t)
    for x in m.ins:
        if x[0] & 0xffff == OP_CONSTANT_COMPOSITE and len(x) > 2: ty.setdefault(x[2], x[1])
    n, out = 0, []
    for x in m.ins:
        if x[0] & 0xffff == OP_IMAGE_FETCH and len(x) > 5 and x[5] & 0x10 and not x[5] & ~0xff:
            k = 6
            for bit, words in IMG_OPERAND_WORDS:
                if bit == 0x10: break
                if x[5] & bit: k += words
            coord, off = x[4], x[k] if k < len(x) else None
            if off is not None and ty.get(coord) is not None and ty.get(coord) == ty.get(off):
                a = m.new_id(); out.append([5 << 16 | OP_IADD, ty[coord], a, coord, off])
                x = x[:4] + [a] + ([x[5] & ~0x10] if x[5] & ~0x10 else []) + x[6:k] + x[k + 1:]
                x[0] = len(x) << 16 | OP_IMAGE_FETCH; n += 1
        out.append(x)
    if n: m.ins[:] = out
    return n


def const_int(m, ident):
    c = m.consts.get(ident)
    return c[1][0] if c and m.int_type(c[0]) else None


def block_rows(m, vid):
    """The 16-byte rows of a uniform block declared as vec4[n] (the translator's constant buffers) that the shader reads: (sorted byte offsets of rows read at
    constant indices, whether any index is computed), from the access chains on the variable"""
    rows, dyn = set(), False
    for x in m.ins:
        if x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and x[3] == vid:
            if len(x) < 6: dyn = True; continue
            r = const_int(m, x[5])
            if r is None: dyn = True
            else: rows.add(r * 16 + (4 * const_int(m, x[6]) if len(x) > 6 and const_int(m, x[6]) is not None else 0))
        elif x[0] & 0xffff == OP_LOAD and x[3] == vid: dyn = True   # the whole block
    return sorted(rows), dyn


CAP_READ_WITHOUT_FORMAT, CAP_WRITE_WITHOUT_FORMAT = 55, 56


def formatless_storage_caps(m):
    """The capabilities a read or write of a storage image declared with format Unknown needs, which the translator leaves out (its images take the format of the answer
    table, and without an answer it declares none). → names of those added (the device features shaderStorageImageRead / WriteWithoutFormat)"""
    imgs = {i for i, x in m.types.items() if x[0] & 0xffff == OP_TYPE_IMAGE and x[7] == 2 and x[8] == 0}   # sampled 2 = storage, format 0 = Unknown
    if not imgs: return []
    typed = {}
    for x in m.ins:
        if x[0] & 0xffff == OP_LOAD and x[1] in imgs: typed[x[2]] = True
    need = set()
    for x in m.ins:
        op = x[0] & 0xffff
        if op == OP_IMAGE_READ and x[3] in typed: need.add(CAP_READ_WITHOUT_FORMAT)
        elif op == OP_IMAGE_WRITE and x[1] in typed: need.add(CAP_WRITE_WITHOUT_FORMAT)
    need -= set(m.caps)
    if not need: return []
    at = max(k for k, y in enumerate(m.ins) if y[0] & 0xffff == OP_CAPABILITY) + 1
    for c in sorted(need): m.ins.insert(at, [2 << 16 | OP_CAPABILITY, c])
    m.index()
    return sorted(CAPS[c] for c in need)


def maybe_ids(x):
    """The words of a function-body instruction that may be ids (the literal operands of the instructions whose literals could pass for an id left out; others whole)"""
    op = x[0] & 0xffff
    if op == OP_EXT_INST: return [x[1], x[3]] + x[5:]
    if op == OP_COMPOSITE_EXTRACT: return x[1:4]
    if op in (OP_COMPOSITE_INSERT, 79): return x[1:5]   # 79: OpVectorShuffle
    if op == 251: return [x[1]] + x[2:3] + x[4::2]   # OpSwitch: selector, default, the targets after each literal
    mask_at = {87: 5, 88: 5, 91: 5, 92: 5, 95: 5, 98: 5, 89: 6, 90: 6, 93: 6, 94: 6, 96: 6, 97: 6, 99: 4}.get(op)   # image sample / fetch / gather / read / write: the image operands mask
    if mask_at is not None and len(x) > mask_at: return x[1:mask_at] + x[mask_at + 1:]
    return x[1:]


SC_PRIVATE = 6
BUILTIN_VIEWPORT_MASK_NV = 5253
CAP_VIEWPORT_MASK_NV, CAP_VIEWPORT_INDEX_LAYER = 5255, 5254


def private_viewport_mask(m):
    """A geometry shader whose passthrough Ryujinx emulates (Program.cs gs_passthrough false: the device has no SPV_NV_geometry_shader_passthrough) keeps NVN's viewport mask
    output (0x3a0) for one use: it emits the primitive only when the mask is not 0 (EmitterContext.PrepareForReturn; a cull, as the hardware sends a primitive with an empty mask
    to no viewport). Without SPV_NV_viewport_array2 (Adreno) the ViewportMaskNV output becomes a Private variable of the same type: the cull stays, the mask leaves the
    shader. What is lost is the routing of a primitive to the viewports the mask names (a single-viewport draw sees no difference). → True when rewritten"""
    var = next((v for v, d in m.decos.items() if d.get(DEC_BUILTIN) == [BUILTIN_VIEWPORT_MASK_NV] and m.vars.get(v, (0, None))[1] == SC_OUTPUT), None)
    if var is None: return False
    ptr, _ = m.vars[var]; t = m.pointee(ptr)[1]
    pptr = m.add_type(OP_TYPE_POINTER, SC_PRIVATE, t)
    chains, in_fn = [], False   # access chains on the variable; any other use in a function body but a load or store is not the shape this rewrite knows
    for x in m.ins:
        op = x[0] & 0xffff
        in_fn = (in_fn or op == OP_FUNCTION) and op != OP_FUNCTION_END
        if op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and x[3] == var:
            if m.pointee(x[1])[0] != SC_OUTPUT: raise ValueError('viewport mask access chain not in Output')
            chains.append(x)
        elif in_fn and var in maybe_ids(x) and op not in (OP_LOAD, OP_STORE): raise ValueError(f'viewport mask used by opcode {op}')
    twins = {}   # Output pointer type of a chain → its Private twin (added after the scan: add_type inserts into m.ins)
    for x in chains:
        if x[1] not in twins: twins[x[1]] = m.add_type(OP_TYPE_POINTER, SC_PRIVATE, m.pointee(x[1])[1])
        x[1] = twins[x[1]]
    keep = []
    for x in m.ins:
        op = x[0] & 0xffff
        if op == OP_DECORATE and x[1] == var: continue
        if op == OP_VARIABLE and x[2] == var: continue   # declared again below, after its new pointer type
        if op == OP_ENTRY_POINT and m.header[1] < 0x10400:   # before SPIR-V 1.4 the interface lists only Input / Output variables (the translator writes 1.3)
            _, k = string_at(x, 3)
            x = x[:k] + [v for v in x[k:] if v != var]; x[0] = len(x) << 16 | OP_ENTRY_POINT
        keep.append(x)
    m.ins = keep; m._insert_decl([4 << 16 | OP_VARIABLE, pptr, var, SC_PRIVATE]); m.index()
    if not any(d.get(DEC_BUILTIN) == [BUILTIN_VIEWPORT_MASK_NV] for d in m.decos.values()) and \
            not any(x[0] & 0xffff == OP_MEMBER_DECORATE and x[3] == DEC_BUILTIN and x[4] == BUILTIN_VIEWPORT_MASK_NV for x in m.ins):
        layer = CAP_VIEWPORT_INDEX_LAYER in m.caps
        m.ins = [x for x in m.ins if not (x[0] & 0xffff == OP_CAPABILITY and x[1] == CAP_VIEWPORT_MASK_NV) and
                 not (x[0] & 0xffff == OP_EXTENSION and not layer and string_at(x, 1)[0] == 'SPV_NV_viewport_array2')]
        m.index()
    return True


OP_CONSTANT_TRUE, OP_CONSTANT_FALSE, OP_SPEC_CONSTANT, OP_CONSTANT_NULL = 41, 42, 50, 46
OP_IEQUAL, OP_FORD_EQUAL, OP_FUNORD_NOT_EQUAL, OP_FORD_LESS, OP_FORD_GREATER, OP_FORD_LESS_EQUAL, OP_FORD_GREATER_EQUAL = 170, 180, 183, 184, 186, 188, 190
OP_SELECTION_MERGE, OP_BRANCH_CONDITIONAL, OP_KILL = 247, 250, 252
DEC_INDEX = 32


def _float_vec(m, t):
    """(component count, float32 type) of a float scalar or vector type, else None"""
    s = m.scalar_of(t)
    if m.op(s) != OP_TYPE_FLOAT or m.types[s][2] != 32: return None
    return (m.types[t][3] if m.op(t) == OP_TYPE_VECTOR else 1), s


def const_written_outputs(m):
    """Colour outputs (locations) every store to which, whole or by component, writes a constant: a clear drawn into an integer target writes 0 that way
    (dmsh_build.py fs_uint's second rule). → the locations, empty unless every output written is such"""
    consts = {x[2] for x in m.ins if x[0] & 0xffff in (OP_CONSTANT, OP_CONSTANT_COMPOSITE, OP_CONSTANT_NULL)}
    outs = m.io(SC_OUTPUT); ptrs = {}
    for loc, (var, _) in outs.items(): ptrs[var] = loc
    for x in m.ins:
        if x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and x[3] in ptrs: ptrs[x[2]] = ptrs[x[3]]
    stored = {}
    for x in m.ins:
        if x[0] & 0xffff == OP_STORE and x[1] in ptrs: stored.setdefault(ptrs[x[1]], []).append(x[2] in consts)
    return set(stored) if stored and all(all(v) for v in stored.values()) else set()


def retype_outputs(m, locs):
    """The fragment outputs at locs, declared float, retyped to the unsigned integer vector of the same width: every store bit-casts its value, every load bit-casts back
    (dmsh_build.py fs_uint: the same bits, nothing converted). An integer attachment takes only an integer output (Vulkan: the numeric class must match). → locations retyped"""
    outs = m.io(SC_OUTPUT); u32 = m.add_type(OP_TYPE_INT, 32, 0)
    want = {}
    for loc in sorted(locs):
        if loc not in outs: continue
        var, t = outs[loc]; fv = _float_vec(m, t)
        if fv: want[var] = (loc, t, fv[0])
    if not want: return []
    def utype(n): return u32 if n == 1 else m.add_type(OP_TYPE_VECTOR, u32, n)
    newptr = {var: m.add_type(OP_TYPE_POINTER, SC_OUTPUT, utype(n)) for var, (_, _, n) in want.items()}
    ptr_of = {var: (t, utype(n)) for var, (_, t, n) in want.items()}   # pointer id → (float value type, uint value type)
    chains = []
    for x in m.ins:
        if x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and x[3] in ptr_of:
            et = m.pointee(x[1])[1]; fv = _float_vec(m, et)
            if not fv: raise ValueError('colour output access chain to a non-float')
            chains.append((x, et, fv[0]))
    for x, et, n in chains:
        ptr_of[x[2]] = (et, utype(n)); x[1] = m.add_type(OP_TYPE_POINTER, SC_OUTPUT, utype(n))
    out, in_fn = [], False
    for x in m.ins:
        op = x[0] & 0xffff
        in_fn = (in_fn or op == OP_FUNCTION) and op != OP_FUNCTION_END
        if op == OP_VARIABLE and x[2] in want: continue   # declared again below, after its new pointer type
        if op == OP_STORE and x[1] in ptr_of:
            v = m.new_id(); out.append([4 << 16 | OP_BITCAST, ptr_of[x[1]][1], v, x[2]]); out.append([3 << 16 | OP_STORE, x[1], v, *x[3:]]); continue
        if op == OP_LOAD and x[3] in ptr_of:
            v = m.new_id(); out.append([x[0], ptr_of[x[3]][1], v, *x[3:]]); out.append([4 << 16 | OP_BITCAST, x[1], x[2], v]); continue
        if op in (OP_STORE, OP_LOAD) or op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN): out.append(x); continue
        if in_fn and any(v in ptr_of for v in maybe_ids(x)):
            raise ValueError(f'colour output used by opcode {op}')
        out.append(x)
    m.ins = out
    for var in want: m._insert_decl([4 << 16 | OP_VARIABLE, newptr[var], var, SC_OUTPUT])
    m.index()
    return sorted(loc for loc, _, _ in want.values())


DEC_PER_PRIMITIVE, CAP_MESH_SHADING = 5271, 5283   # PerPrimitiveEXT, MeshShadingEXT (SPV_EXT_mesh_shader)


def per_primitive_inputs(m, locs):
    """The inputs of a pixel shader that a mesh program writes per primitive (shader_split.py mesh_facts: the flat inputs of the mesh's pixel shader): Vulkan matches a
    PerPrimitiveEXT mesh output only with an input decorated the same, so each such input gets PerPrimitiveEXT, and the module the capability and extension. Done only for a
    shader drawn with mesh programs alone (vk_android_build.py MESH_PRIM): it is not valid in a vertex pipeline any more. → the locations decorated"""
    ins = m.io(SC_INPUT)
    done = sorted(l for l in locs if l in ins)
    if not done: return []
    at = next(k for k, y in enumerate(m.ins) if y[0] & 0xffff == OP_DECORATE)
    for l in done: m.ins.insert(at, [3 << 16 | OP_DECORATE, ins[l][0], DEC_PER_PRIMITIVE])
    if CAP_MESH_SHADING not in m.caps:
        k = max(k for k, y in enumerate(m.ins) if y[0] & 0xffff == OP_CAPABILITY)
        m.ins.insert(k + 1, [2 << 16 | OP_CAPABILITY, CAP_MESH_SHADING])
    if 'SPV_EXT_mesh_shader' not in m.exts:
        k = max(k for k, y in enumerate(m.ins) if y[0] & 0xffff in (OP_CAPABILITY, OP_EXTENSION))
        e = lit_string('SPV_EXT_mesh_shader'); m.ins.insert(k + 1, [(1 + len(e)) << 16 | OP_EXTENSION, *e])
    if m.header[1] < 0x10400:   # the extension needs SPIR-V 1.4, whose entry point lists every global variable it uses, not only Input / Output: all of them are listed
        m.header[1] = 0x10400
        have = set(m.interface())
        glob = [x[2] for x in m.ins if x[0] & 0xffff == OP_VARIABLE and x[3] != SC_FUNCTION and x[2] not in have]
        k = m.ins.index(m.entry)
        m.ins[k] = m.entry + glob; m.ins[k][0] = len(m.ins[k]) << 16 | OP_ENTRY_POINT
        m.index(); buffer_block_to_storage(m)
    m.index()
    return done


OP_PTR_ACCESS_CHAIN, DEC_BLOCK_ = 67, 2


def buffer_block_to_storage(m):
    """SPIR-V 1.4 has no BufferBlock: a Uniform-class block decorated BufferBlock (this translator's storage buffers) becomes a StorageBuffer-class Block, and every pointer
    into it (its variables, the access chains from them) gets a StorageBuffer pointer type of the same pointee. → the number of blocks rewritten"""
    blocks = {x[1] for x in m.ins if x[0] & 0xffff == OP_DECORATE and x[2] == DEC_BUFFER_BLOCK}
    if not blocks: return 0
    def sb_ptr(uptr):   # the StorageBuffer pointer of a Uniform pointer type's pointee, declared right after it (a global variable may use it)
        t = m.types[uptr][3]
        i = m.find_type(OP_TYPE_POINTER, SC_STORAGE_BUFFER, t)
        if i is not None: return i
        i = m.new_id(); x = [4 << 16 | OP_TYPE_POINTER, i, SC_STORAGE_BUFFER, t]
        m.ins.insert(m.ins.index(m.types[uptr]) + 1, x); m.types[i] = x
        return i
    def elem(t):   # an array of blocks (a descriptor array): its element
        while m.op(t) in (OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY): t = m.types[t][2]
        return t
    ptrs = {i: sb_ptr(i) for i, x in list(m.types.items()) if x[0] & 0xffff == OP_TYPE_POINTER and x[2] == SC_UNIFORM and elem(x[3]) in blocks}
    sb = set()
    for x in list(m.ins):   # a copy: sb_ptr may insert a pointer type
        op = x[0] & 0xffff
        if op == OP_DECORATE and x[1] in blocks and x[2] == DEC_BUFFER_BLOCK: x[2] = DEC_BLOCK_
        elif op == OP_VARIABLE and x[1] in ptrs and x[3] == SC_UNIFORM: x[1] = ptrs[x[1]]; x[3] = SC_STORAGE_BUFFER; sb.add(x[2])
        elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN, OP_PTR_ACCESS_CHAIN, OP_COPY_OBJECT) and x[3] in sb:
            x[1] = sb_ptr(x[1]); sb.add(x[2])
    m.index()
    return len(blocks)


def dual_source(m):
    """Dual-source blending (dmsh_build.py fs_dual): NVN's SRC1 factors read output 1 as target 0's second source; Vulkan reads it only from Location 0 Index 1. For a shader
    writing exactly colours 0 and 1: output 1 moved to Location 0, Index 1. → True when rewritten"""
    outs = m.io(SC_OUTPUT)
    if sorted(outs) != [0, 1]: return False
    var = outs[1][0]
    for k, x in enumerate(m.ins):
        if x[0] & 0xffff == OP_DECORATE and x[1] == var and x[2] == DEC_LOCATION:
            x[3] = 0; m.ins.insert(k + 1, [4 << 16 | OP_DECORATE, var, DEC_INDEX, 1]); break
    m.index()
    return True


ALPHA_CMP = {2: OP_FORD_LESS, 3: OP_FORD_EQUAL, 4: OP_FORD_LESS_EQUAL, 5: OP_FORD_GREATER, 6: OP_FUNORD_NOT_EQUAL, 7: OP_FORD_GREATER_EQUAL}   # NVN_ALPHA_FUNC_* → comparison of alpha with the reference


def alpha_test(m, spec_id):
    """NVN's alpha test (ColorStateSetAlphaTest / SetAlphaRef; dmsh_alpha.py on Metal): colour 0's alpha compared with a reference before blending, the fragment dropped when the
    comparison fails. The function is a uint specialization constant spec_id (NVN_ALPHA_FUNC_*: 1 never, 2 less … 7 gequal, 8 always; default 0 = no test, which the driver
    folds away), part of the pipeline as the colour state is; the reference is per draw, the float bits of the support buffer's word 0 (Ryujinx's unused alpha_test field), so a
    reference that changes every draw (a dissolve) keeps the pipeline. The test runs before every return of the entry function. → True when added (a float vec4 colour 0 and
    the support buffer present)"""
    outs = m.io(SC_OUTPUT)
    if 0 not in outs: return False
    c0, t = outs[0]; fv = _float_vec(m, t)
    if not fv or fv[0] != 4: return False
    f32 = fv[1]
    sb = next((v for v, (p, sc) in m.vars.items() if sc == SC_UNIFORM and m.names.get(v) == 'support_buffer'), None)
    if sb is None: return False
    st = m.pointee(m.vars[sb][0])[1]
    if m.op(st) != OP_TYPE_STRUCT or m.int_type(m.types[st][2]) != (32, 0): return False
    u32 = m.types[st][2]; i32 = m.find_type(OP_TYPE_INT, 32, 1) or m.add_type(OP_TYPE_INT, 32, 1)
    boolean = m.add_type(OP_TYPE_BOOL); pu = m.add_type(OP_TYPE_POINTER, SC_UNIFORM, u32)
    zero = m.add_const(i32, 0)
    fconst = {n: m.add_const(u32, n) for n in [1] + list(ALPHA_CMP)}
    spec = m.new_id(); m._insert_decl([4 << 16 | OP_SPEC_CONSTANT, u32, spec, 0])
    t_ = next((x[2] for x in m.ins if x[0] & 0xffff == OP_CONSTANT_TRUE and x[1] == boolean), None)
    if t_ is None: t_ = m.new_id(); m._insert_decl([3 << 16 | OP_CONSTANT_TRUE, boolean, t_])
    f_ = next((x[2] for x in m.ins if x[0] & 0xffff == OP_CONSTANT_FALSE and x[1] == boolean), None)
    if f_ is None: f_ = m.new_id(); m._insert_decl([3 << 16 | OP_CONSTANT_FALSE, boolean, f_])
    at = next(k for k, y in enumerate(m.ins) if y[0] & 0xffff == OP_DECORATE)
    m.ins.insert(at, [4 << 16 | OP_DECORATE, spec, DEC_SPEC_ID, spec_id])
    name = lit_string('dm_alpha_func'); m.ins.insert(next(k for k, y in enumerate(m.ins) if y[0] & 0xffff in (OP_NAME, OP_DECORATE)), [(2 + len(name)) << 16 | OP_NAME, spec, *name])
    vt = t; a, e = m.function_range(m.entry_function()); out, n = [], 0
    for k, x in enumerate(m.ins):
        if a < k < e and x[0] & 0xffff == OP_RETURN:
            col, al, pw, rw, ref = (m.new_id() for _ in range(5))
            out += [[4 << 16 | OP_LOAD, vt, col, c0], [5 << 16 | OP_COMPOSITE_EXTRACT, f32, al, col, 3],
                    [5 << 16 | OP_ACCESS_CHAIN, pu, pw, sb, zero], [4 << 16 | OP_LOAD, u32, rw, pw], [4 << 16 | OP_BITCAST, f32, ref, rw]]
            res = t_   # an unknown function (0, 8 always) passes
            for fn in [1] + list(ALPHA_CMP):
                is_fn, cmp_, r2 = m.new_id(), m.new_id(), m.new_id()
                out.append([5 << 16 | OP_IEQUAL, boolean, is_fn, spec, fconst[fn]])
                if fn == 1: cmp_ = f_
                else: out.append([5 << 16 | ALPHA_CMP[fn], boolean, cmp_, al, ref])
                out.append([6 << 16 | OP_SELECT, boolean, r2, is_fn, cmp_, res]); res = r2
            ok, kill = m.new_id(), m.new_id()
            out += [[3 << 16 | OP_SELECTION_MERGE, ok, 0], [4 << 16 | OP_BRANCH_CONDITIONAL, res, ok, kill],
                    [2 << 16 | OP_LABEL, kill], [1 << 16 | OP_KILL], [2 << 16 | OP_LABEL, ok]]
            n += 1
        out.append(x)
    m.ins = out; m.index()
    return n > 0


OP_FUNCTION_PARAMETER, OP_FUNCTION_CALL = 55, 57
OP_IADD, OP_ISUB, OP_IMUL, OP_SHR_LOGICAL, OP_SHR_ARITH, OP_SHL_LOGICAL, OP_BITWISE_OR, OP_BITWISE_XOR, OP_BITWISE_AND = 128, 130, 132, 194, 195, 196, 197, 198, 199
OP_CONTROL_BARRIER, OP_GROUP_NON_UNIFORM_SHUFFLE = 224, 345
SC_WORKGROUP, SCOPE_SUBGROUP, SEM_ACQ_REL_WORKGROUP = 4, 3, 0x8 | 0x100
BUILTIN_SUBGROUP_SIZE, BUILTIN_SUBGROUP_ID, BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID = 36, 40, 41
CAP_GROUP_NON_UNIFORM, EXEC_MODE_LOCAL_SIZE = 61, 17
VN_OPS = {OP_IADD, OP_ISUB, OP_IMUL, OP_SHR_LOGICAL, OP_SHR_ARITH, OP_SHL_LOGICAL, OP_BITWISE_OR, OP_BITWISE_XOR, OP_BITWISE_AND}


def input_builtin(m, builtin, ty):
    """The Input variable of a builtin (a plain variable), added with its decoration and in the entry point's interface when the module has none → its id"""
    v, member = m.builtin_var(SC_INPUT, builtin)
    if v is not None and member is None: return v
    p = m.add_type(OP_TYPE_POINTER, SC_INPUT, ty); v = m.new_id()
    m._insert_decl([4 << 16 | OP_VARIABLE, p, v, SC_INPUT])
    m.ins.insert(next(k for k, x in enumerate(m.ins) if x[0] & 0xffff == OP_DECORATE), [4 << 16 | OP_DECORATE, v, DEC_BUILTIN, builtin])
    k = next(k for k, x in enumerate(m.ins) if x[0] & 0xffff == OP_ENTRY_POINT)
    m.ins[k] = m.ins[k] + [v]; m.ins[k][0] = len(m.ins[k]) << 16 | OP_ENTRY_POINT
    m.index()
    return v


def workgroup_bytes(m):
    """Bytes of the module's Workgroup variables (arrays of 32-bit scalars or vectors, as the translator declares them); None for another type"""
    n = 0
    for vid, (ptr, sc) in m.vars.items():
        if sc != SC_WORKGROUP: continue
        t, count = m.strip_arrays(m.pointee(ptr)[1])
        comps = m.types[t][3] if m.op(t) == OP_TYPE_VECTOR else 1
        s = m.scalar_of(t)
        if m.op(s) not in (OP_TYPE_INT, OP_TYPE_FLOAT) or m.types[s][2] != 32 or not count: return None
        n += 4 * comps * count
    return n


class BlockValues:
    """Value numbers within a block of the translator's code, which keeps its registers in Function variables: a load of a variable numbers as the value last stored to it in
    the block, or as (variable, version) with a version that every store and every call taking it as an argument moves; bit casts and copies keep their operand's number;
    integer arithmetic numbers by its operation and operands' numbers, a load of an Input variable (or of a constant index into one) by the variable. Two values with one
    number hold the same bits in every lane. see(x) before x is passed on; block()
    at each label; function() at each OpFunction"""
    def __init__(self, m): self.m = m; self.inptr = {}; self.function()
    def function(self): self.fvars, self.fbase, self.ver, self.stored, self.vn = set(), {}, {}, {}, {}
    def block(self): self.stored = {}
    def __call__(self, i): return self.vn.get(i, i)
    def key_of_var(self, p):   # the number of what a load of Function variable p would give here
        b = self.fbase.get(p, p)
        return self(self.stored[p]) if p in self.stored else ('ld', b, self.ver.get(b, 0))
    def bump(self, p):
        b = self.fbase.get(p, p); self.ver[b] = self.ver.get(b, 0) + 1; self.stored.pop(b, None)
    def see(self, x):
        m, op = self.m, x[0] & 0xffff
        if op == OP_FUNCTION: self.function()
        elif op == OP_LABEL: self.block()
        elif (op == OP_VARIABLE and len(x) > 3 and x[3] == SC_FUNCTION) or (op == OP_FUNCTION_PARAMETER and m.op(x[1]) == OP_TYPE_POINTER and m.types[x[1]][2] == SC_FUNCTION):
            self.fvars.add(x[2])   # a local variable, or the translator's pointer argument (written only through a store here or by a call)
        elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and (x[3] in self.fvars or x[3] in self.fbase): self.fbase[x[2]] = self.fbase.get(x[3], x[3])
        elif op == OP_STORE and (x[1] in self.fvars or x[1] in self.fbase):
            self.bump(x[1])
            if x[1] in self.fvars: self.stored[x[1]] = x[2]
        elif op == OP_LOAD and x[3] in self.fvars: self.vn[x[2]] = self.key_of_var(x[3])
        elif op == OP_LOAD and (x[3] in self.inptr or m.vars.get(x[3], (0, 0))[1] == SC_INPUT): self.vn[x[2]] = ('in', self.inptr.get(x[3], x[3]))
        elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and m.vars.get(x[3], (0, 0))[1] == SC_INPUT and all(i in m.consts for i in x[4:]): self.inptr[x[2]] = (x[3], *x[4:])
        elif op in (OP_BITCAST, OP_COPY_OBJECT): self.vn[x[2]] = self(x[3])
        elif op in VN_OPS: self.vn[x[2]] = (op, self(x[3]), self(x[4]))
        elif op == OP_FUNCTION_CALL:
            for p in x[4:]:
                if p in self.fvars or p in self.fbase: self.bump(p)

def shuffle_lds(m, shared_max):
    """The 32-bit subgroup shuffles of a compute module through workgroup memory (the library's DMSH_VK_MOD_SHUFFLE_LDS module, for drivers without an arbitrary-index
    shuffle instruction): each lane stores the value at its slot of its subgroup's row (SubgroupId * SubgroupSize + lane), a subgroup barrier, the slot of the lane asked
    for is read. Turnip has no such instruction: ir3 makes each OpGroupNonUniformShuffle a loop of rotates, one pass per distinct (index - lane) in the subgroup, up to 24
    for the translator's LDSM / MOVM gathers (agl's upscaler network on Donkey Kong Bananza: 389 loops, 10.7 ms a dispatch on the Adreno 840 against 2.0 ms this way);
    Qualcomm's driver shuffles natively and is faster with its base module (1.7 ms against 1.9). Two rows alternate, so a store comes after the other row's reads only
    with the barrier of the exchange between them; an exchange of the value a row still holds in the same block reads it without a store or barrier (LDSM's four reads of
    one address, MOVM's two); the last exchange before the block ends or another function is called gets a trailing barrier. The module is rewritten in place.
    → bytes of workgroup memory added (0: not a compute module with such shuffles, or over shared_max with them; the module is then unchanged)"""
    if m.model != 'cs': return 0
    shuf = [x for x in m.ins if x[0] & 0xffff == OP_GROUP_NON_UNIFORM_SHUFFLE and m.op(x[1]) in (OP_TYPE_INT, OP_TYPE_FLOAT) and m.types[x[1]][2] == 32
            and m.consts.get(x[3], (0, [0]))[1] == [SCOPE_SUBGROUP]]
    ls = next((x[1:4] for x in m.modes if x[0] == EXEC_MODE_LOCAL_SIZE), None)   # (m.modes: the entry point's, without its id)
    used = workgroup_bytes(m)
    if not shuf or ls is None or used is None: return 0
    words = -(-ls[0] * ls[1] * ls[2] // 128) * 128   # a row: SubgroupId * SubgroupSize + lane, subgroups of up to 128 lanes
    if used + 8 * words > shared_max: return 0
    u32 = m.add_type(OP_TYPE_INT, 32, 0)
    arr = m.add_type(OP_TYPE_ARRAY, u32, m.add_const(u32, 2 * words))
    p_arr = m.add_type(OP_TYPE_POINTER, SC_WORKGROUP, arr); p_u = m.add_type(OP_TYPE_POINTER, SC_WORKGROUP, u32)
    xs = m.new_id(); m._insert_decl([4 << 16 | OP_VARIABLE, p_arr, xs, SC_WORKGROUP])
    row_off = [None, m.add_const(u32, words)]; C = {v: m.add_const(u32, v) for v in (SCOPE_SUBGROUP, SEM_ACQ_REL_WORKGROUP)}
    sgid, sgsz, sgl = (input_builtin(m, b, u32) for b in (BUILTIN_SUBGROUP_ID, BUILTIN_SUBGROUP_SIZE, BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID))
    if CAP_GROUP_NON_UNIFORM not in m.caps:
        at = max(k for k, x in enumerate(m.ins) if x[0] & 0xffff == OP_CAPABILITY); m.ins.insert(at + 1, [2 << 16 | OP_CAPABILITY, CAP_GROUP_NON_UNIFORM])
    nm = lit_string('dm_shuffle_rows'); m.ins.insert(next(k for k, y in enumerate(m.ins) if y[0] & 0xffff in (OP_NAME, OP_DECORATE)), [(2 + len(nm)) << 16 | OP_NAME, xs, *nm])
    barrier = [4 << 16 | OP_CONTROL_BARRIER, C[SCOPE_SUBGROUP], C[SCOPE_SUBGROUP], C[SEM_ACQ_REL_WORKGROUP]]
    m.index()
    want = {id(x) for x in shuf}
    calls, cur = {}, None   # functions that exchange, themselves or through a call: a call of one ends the caller's run of exchanges
    for x in m.ins:
        op = x[0] & 0xffff
        if op == OP_FUNCTION: cur = x[2]; calls[cur] = [False, set()]
        elif op == OP_GROUP_NON_UNIFORM_SHUFFLE and id(x) in want: calls[cur][0] = True
        elif op == OP_FUNCTION_CALL: calls[cur][1].add(x[3])
    exch = {f for f, (h, _) in calls.items() if h}
    while True:
        more = {f for f, (_, c) in calls.items() if f not in exch and c & exch}
        if not more: break
        exch |= more
    out, trail = [], []   # trail: positions in out after an exchange's read that take a barrier
    V = BlockValues(m)
    rows, dirty, last = [None, None], [False, False], None   # the value each row holds in this block, rows read since the last barrier, the last exchange's read
    def close():   # the block ends or a function that exchanges runs: the last exchange gets a trailing barrier, and the rows' contents count as unknown after it
        nonlocal rows, dirty, last
        if last is not None: trail.append(last)
        rows, dirty, last = [None, None], [False, False], None
    def em(op, ty, *ops):
        r = m.new_id(); out.append([(3 + len(ops)) << 16 | op, ty, r, *ops]); return r
    for x in m.ins:
        op = x[0] & 0xffff
        if op in (OP_LABEL, OP_FUNCTION_END) or (op == OP_FUNCTION_CALL and x[3] in exch): close()
        V.see(x)
        if op != OP_GROUP_NON_UNIFORM_SHUFFLE or id(x) not in want: out.append(x); continue
        key = V(x[4])   # (a value number: the same in two shuffles of this block only when every lane's value is the same)
        row = next((r for r in (0, 1) if rows[r] == key), None)
        sg = em(OP_IMUL, u32, em(OP_LOAD, u32, sgid), em(OP_LOAD, u32, sgsz))
        if row is None:   # store into a row nothing has read since the last barrier
            clean = [r for r in (0, 1) if not dirty[r]]
            if not clean: trail.append(last); clean = [0, 1]
            row = next((r for r in clean if rows[r] is None), clean[0])
            base = sg if row == 0 else em(OP_IADD, u32, sg, row_off[1])
            v = x[4] if x[1] == u32 else em(OP_BITCAST, u32, x[4])
            out.append([3 << 16 | OP_STORE, em(OP_ACCESS_CHAIN, p_u, xs, em(OP_IADD, u32, base, em(OP_LOAD, u32, sgl))), v])
            out.append(list(barrier))
            rows[row] = key; dirty = [False, False]   # the barrier orders every earlier read
        else: base = sg if row == 0 else em(OP_IADD, u32, sg, row_off[1])
        dirty[row] = True
        r = em(OP_LOAD, u32, em(OP_ACCESS_CHAIN, p_u, xs, em(OP_IADD, u32, base, x[5])))
        out.append([4 << 16 | (OP_COPY_OBJECT if x[1] == u32 else OP_BITCAST), x[1], x[2], r])
        last = len(out)
    for k in sorted(set(trail), reverse=True): out.insert(k, list(barrier))
    m.ins = out; m.index()
    return 8 * words


HMMA_NAME = 'dmsh_hmma16816_f16'
OP_MEMORY_BARRIER, OP_COMPOSITE_CONSTRUCT, OP_TYPE_FUNCTION_ = 225, 80, 33
GLSL_FMA, GLSL_PACK_HALF_2X16, GLSL_UNPACK_HALF_2X16 = 50, 58, 62
NO_RESULT_OPS = {OP_STORE, OP_CONTROL_BARRIER, OP_MEMORY_BARRIER, OP_RETURN, OP_RETURN_VALUE, 246, 247, 249, 250, 251, 252, OP_FUNCTION_END}


def hmma_helper(m):
    """The translator's HMMA.16816 helper (HelperFunctionManager.GenerateHmma16816F16) when it has the translator's shape: 11 pointer arguments, one block, 8 + 64 accesses
    of one Workgroup array of u32 used by nothing else, two subgroup barriers → {fid, i32, u32, scratch, barrier}, else None"""
    fid = functions_named(m).get(HMMA_NAME)
    if fid is None: return None
    a, e = m.function_range(fid)
    i32 = m.ins[a][1]
    if m.int_type(i32) != (32, 1): return None
    params = [x for x in m.ins[a + 1:e] if x[0] & 0xffff == OP_FUNCTION_PARAMETER]
    if len(params) != 11 or any(m.types.get(p[1], [0, 0, 0, 0])[2:4] != [SC_FUNCTION, i32] for p in params): return None
    body = m.ins[a + 1 + len(params):e]
    if sum(1 for x in body if x[0] & 0xffff == OP_LABEL) != 1: return None
    chains = [x for x in body if x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN)]
    if len(chains) != 8 + 64 or len({x[3] for x in chains}) != 1: return None
    scratch = chains[0][3]; pwu = chains[0][1]
    if m.vars.get(scratch, (0, 0))[1] != SC_WORKGROUP or m.types.get(pwu, [0, 0, 0])[2] != SC_WORKGROUP: return None
    u32 = m.pointee(pwu)[1]; arr = m.pointee(m.vars[scratch][0])[1]
    if m.int_type(u32) != (32, 0) or m.op(arr) != OP_TYPE_ARRAY or m.types[arr][2] != u32 or m.consts[m.types[arr][3]][1][0] % 4: return None
    if any(scratch in x[1:] for k, x in enumerate(m.ins) if not (a <= k <= e) and not (x[0] & 0xffff in (OP_NAME, OP_DECORATE, OP_VARIABLE) or 19 <= x[0] & 0xffff <= 52)):
        return None   # the scratch array is the helper's alone
    groups, cur = [], None   # runs of barrier instructions: two equal ones (the translator's subgroupBarrier, or a memory barrier and an execution barrier)
    for x in body:
        if x[0] & 0xffff in (OP_CONTROL_BARRIER, OP_MEMORY_BARRIER):
            if cur is None: cur = []; groups.append(cur)
            cur.append(list(x))
        else: cur = None
    if len(groups) != 2 or groups[0] != groups[1]: return None
    return {'fid': fid, 'i32': i32, 'u32': u32, 'scratch': scratch, 'barrier': groups[0]}


def hmma_lds(m):
    """Every call of the translator's HMMA.16816 helper (f16 sources and accumulator) replaced at its call site, with the gathers of A and B shared by the calls of a block
    that take the same fragments. The translator's body stores the lane's 8 fragment words into its warp's 256-word block of the scratch array, then reads 64 words one by one
    (both rows of A, both columns of B, for each output) and converts each half: 240 calls in agl's upscaler network on Donkey Kong Bananza, 35 ms a dispatch on the Adreno
    840 with Qualcomm's driver, 29 ms on Turnip (with its shuffles). Here the block is uvec4 rows: A row-major (16 rows of 8 words) and B transposed (8 rows), each row's k
    words in the order (k-word t, t + 4) for t = 0..3 so a lane's two words of a row are adjacent; C stays in the lane, which is the only one to need it. A gather is the
    stores, a subgroup barrier, 8 uvec4 reads (the lane's two rows of A and two columns of B, converted to f32 once), a barrier; a call whose A (or B) fragments are those of
    an earlier call in the block (the same value numbers, BlockValues) reuses its rows: agl's network multiplies two A tiles by two B tiles, so 8 calls gather 4 A and 6 B.
    The arithmetic is the translator's (f32 FMAs from C over k in order, one rounding to f16), so D is bit for bit the same. 8.0 ms instead of 35 on Qualcomm's driver,
    10.7 ms instead of 29 on Turnip (with shuffle_lds; 13 ms with the translator's body). The helper function is removed.
    → number of calls lowered (0: no helper of the translator's shape; the module is then unchanged)"""
    h = hmma_helper(m)
    if h is None: return 0
    fid, i32, u32, scratch = h['fid'], h['i32'], h['u32'], h['scratch']
    glsl = ensure_glsl(m)
    f32 = m.add_type(OP_TYPE_FLOAT, 32); v2f = m.add_type(OP_TYPE_VECTOR, f32, 2); uv4 = m.add_type(OP_TYPE_VECTOR, u32, 4)
    n = m.consts[m.types[m.pointee(m.vars[scratch][0])[1]][3]][1][0]
    arr4 = m.add_type(OP_TYPE_ARRAY, uv4, m.add_const(u32, n // 4))
    p_arr4 = m.add_type(OP_TYPE_POINTER, SC_WORKGROUP, arr4); p_uv4 = m.add_type(OP_TYPE_POINTER, SC_WORKGROUP, uv4); p_u = m.add_type(OP_TYPE_POINTER, SC_WORKGROUP, u32)
    k = next(k for k, x in enumerate(m.ins) if x[0] & 0xffff == OP_VARIABLE and x[2] == scratch)
    x = m.ins.pop(k); x[1] = p_arr4; m._insert_decl(x)   # the scratch array as uvec4s, declared after its new type
    CI = {v: m.add_const(i32, v) for v in (1, 2, 3, 8, 32)}
    CU = {v: m.add_const(u32, v) for v in (0, 1)}
    m.index()
    calls = 0
    for f in [x[2] for x in m.ins if x[0] & 0xffff == OP_FUNCTION and x[2] != fid]:
        a, e = m.function_range(f)
        out = []; V = BlockValues(m); cache = {}
        def em(op, ty, *ops):
            r = m.new_id(); out.append([(3 + len(ops)) << 16 | op, ty, r, *ops]); return r
        def I(op, p, q): return em(op, i32, p, q)
        for x in m.ins[a:e + 1]:
            op = x[0] & 0xffff
            if op == OP_LABEL: cache = {}
            if not (op == OP_FUNCTION_CALL and x[3] == fid): V.see(x); out.append(x); continue
            # [OpFunctionCall, i32, result, fid, lane, base, a0..a3, b0, b1, c0, c1, d1 (out)]
            args = x[4:15]
            key = [V.key_of_var(p) for p in args[:10]]
            val = [em(OP_LOAD, i32, p) for p in args[:10]]
            lane, bse = val[0], val[1]
            g = I(OP_SHR_LOGICAL, lane, CI[2]); t = I(OP_BITWISE_AND, lane, CI[3])
            kA = ('A', *key[:6]); kB = ('B', *key[:2], *key[6:8])
            need = [q for q in (kA, kB) if q not in cache]
            if need:
                b4 = I(OP_SHR_LOGICAL, bse, CI[2])   # the warp's block in uvec4s: A rows at 2 r, B rows at 32 + 2 n
                vi = I(OP_SHR_LOGICAL, t, CI[1]); c0 = em(OP_BITCAST, u32, I(OP_SHL_LOGICAL, I(OP_BITWISE_AND, t, CI[1]), CI[1])); c1 = em(OP_IADD, u32, c0, CU[1])
                rowA = lambda r: I(OP_IADD, b4, I(OP_SHL_LOGICAL, r, CI[1]))
                rowB = lambda c: I(OP_IADD, b4, I(OP_IADD, CI[32], I(OP_SHL_LOGICAL, c, CI[1])))
                def st(row, w0, w1):   # the lane's k-words t and t + 4 of a row: components 2 (t & 1) and + 1 of its uvec4 t / 2
                    v4 = I(OP_IADD, row, vi)
                    for c, w in ((c0, w0), (c1, w1)): out.append([3 << 16 | OP_STORE, em(OP_ACCESS_CHAIN, p_u, scratch, v4, c), em(OP_BITCAST, u32, w)])
                g8 = I(OP_IADD, g, CI[8])
                if kA in need: st(rowA(g), val[2], val[4]); st(rowA(g8), val[3], val[5])
                if kB in need: st(rowB(g), val[6], val[7])
                out.extend(list(b) for b in h['barrier'])
                def ld_row(r0):   # a row's 8 words, read as two uvec4, converted to vec2 (lo, hi halves) in k order
                    w = []
                    for p in range(2):
                        v4 = em(OP_LOAD, uv4, em(OP_ACCESS_CHAIN, p_uv4, scratch, I(OP_IADD, r0, CI[1]) if p else r0))
                        w += [em(OP_COMPOSITE_EXTRACT, u32, v4, c) for c in range(4)]
                    return [em(OP_EXT_INST, v2f, glsl, GLSL_UNPACK_HALF_2X16, w[2 * (kw % 4) + kw // 4]) for kw in range(8)]
                if kA in need: cache[kA] = [ld_row(rowA(g)), ld_row(rowA(g8))]
                if kB in need:
                    t2 = I(OP_SHL_LOGICAL, t, CI[1]); cache[kB] = [ld_row(rowB(t2)), ld_row(rowB(I(OP_IADD, t2, CI[1])))]
                out.extend(list(b) for b in h['barrier'])
            A, B = cache[kA], cache[kB]
            d = []
            for q in range(2):   # D rows g, g + 8: the translator's order, C then k = 0..15 per output
                cv = em(OP_EXT_INST, v2f, glsl, GLSL_UNPACK_HALF_2X16, em(OP_BITCAST, u32, val[8 + q]))
                acc = [em(OP_COMPOSITE_EXTRACT, f32, cv, 0), em(OP_COMPOSITE_EXTRACT, f32, cv, 1)]
                for j in range(2):
                    for kw in range(8):
                        for hh in range(2):
                            acc[j] = em(OP_EXT_INST, f32, glsl, GLSL_FMA, em(OP_COMPOSITE_EXTRACT, f32, A[q][kw], hh), em(OP_COMPOSITE_EXTRACT, f32, B[j][kw], hh), acc[j])
                d.append(em(OP_BITCAST, i32, em(OP_EXT_INST, u32, glsl, GLSL_PACK_HALF_2X16, em(OP_COMPOSITE_CONSTRUCT, v2f, *acc))))
            st_out = [3 << 16 | OP_STORE, args[10], d[1]]
            out.append(st_out); V.see(st_out)
            out.append([4 << 16 | OP_COPY_OBJECT, i32, x[2], d[0]])
            calls += 1
        m.ins[a:e + 1] = out
    a, e = m.function_range(fid)   # the helper, no longer called
    gone = {y[2] for y in m.ins[a:e + 1] if len(y) > 2 and (y[0] & 0xffff) not in NO_RESULT_OPS} | {y[1] for y in m.ins[a:e + 1] if y[0] & 0xffff == OP_LABEL}
    del m.ins[a:e + 1]
    m.ins = [y for y in m.ins if not (y[0] & 0xffff in (OP_NAME, OP_DECORATE) and y[1] in gone)]
    m.index()
    return calls


# ---- raw global memory helpers without branches (gmem_flatten)
OP_UNDEF, OP_PHI, OP_LOOP_MERGE, OP_BRANCH, OP_SWITCH, OP_UNREACHABLE = 1, 245, 246, 249, 251, 255
OP_CONVERT_U_TO_PTR, OP_LOGICAL_AND, OP_LOGICAL_NOT = 120, 167, 168
OP_ATOMIC_COMPARE_EXCHANGE, OP_ATOMIC_UMAX = 230, 239
ATOMIC_OPS = set(range(227, 243)) | {5614, 5615, 6035}   # OpAtomicLoad .. OpAtomicXor, OpAtomicFMinEXT / FMaxEXT, OpAtomicFAddEXT


def const_bool(m, value):
    b = m.add_type(OP_TYPE_BOOL); op = OP_CONSTANT_TRUE if value else OP_CONSTANT_FALSE
    for x in m.ins:
        if x[0] & 0xffff == op and x[1] == b: return x[2]
    c = m.new_id(); m._insert_decl([3 << 16 | op, b, c]); return c


def gmem_flatten(m, map_var, log_var, sink_word):
    """The translator's raw global memory helpers (GlobalToStorage.TryGenerateRawStorageOp: an address the shader loaded, found in the host's memory pool table, accessed
    through a physical storage buffer pointer, a store's page put in the write log) made one block. Their branches, the pool check around the access and the write log's 8
    probes with an early exit each, made an inlined store 9 nested ifs: Ocarina of Time's video network (NDVE) has 64 to 384 such stores and loads in a dispatch, and
    Turnip's spiller keeps two words for every value in every block (ir3_spill.c compute_next_distance), 2 GB for one of its programs (1,768 blocks); with the
    if-clones of nir_opt_copy_prop_vars over the translator's temporaries, the largest went past the phone's memory. Here each instruction of a helper runs under the
    condition of the path that held it (and that no return was taken before it): a store to a function variable keeps the old value where the condition is false
    (select), an access through a physical pointer goes to the sink (the host's buffer whose address entry 0 of the table holds, dmsh_abi_vk.h) where it is false, a write
    log probe compares with ~0 (a page is below 2^31, so no slot holds it: no write) and the log's overflow flag is raised with an atomic max of 1 or 0; the returned value is
    selected. The accesses and probes the helper made stay the same; a skipped access now reaches the sink, a skipped probe is an atomic that writes nothing.
    A helper of another shape (a loop: the small-integer store's compare-and-swap; a call, a barrier, a phi, another side effect) is left as it is.
    map_var / log_var: the table's and the log's variables (vk_android_build.py storage_slots: KIND_GMEM_MAP / KIND_GMEM_LOG), log_var None for a shader that only reads;
    sink_word: the table word where the sink's address starts (dmsh_abi_vk.h DMSH_VK_GMEM_SINK_WORD, low then high).
    → number of helpers made one block"""
    if map_var is None: return 0
    entry = m.entry_function()
    funcs = [x[2] for x in m.ins if x[0] & 0xffff == OP_FUNCTION and x[2] != entry]
    done = 0
    for f in funcs:
        a, e = m.function_range(f)
        body = m.ins[a:e + 1]
        if not any(x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and x[3] == map_var for x in body): continue
        if not any(x[0] & 0xffff == OP_CONVERT_U_TO_PTR for x in body): continue
        new = _flatten_helper(m, body, map_var, log_var, sink_word)
        if new is None: continue
        a, e = m.function_range(f)   # (the new types and constants went in before the functions)
        m.ins[a:e + 1] = new
        done += 1
    if done: m.index()
    return done


class _Skip(Exception): pass


def _flatten_helper(m, body, map_var, log_var, sink_word):
    try: return _flatten(m, body, map_var, log_var, sink_word)
    except _Skip: return None


def _flatten(m, body, map_var, log_var, sink_word):
    fn = body[0]; ret_t = fn[1]
    head = [x for x in body[1:] if x[0] & 0xffff == OP_FUNCTION_PARAMETER]
    # blocks: label → (instructions, merge, terminator)
    blocks, order, cur, ins = {}, [], None, []
    for x in body[1 + len(head):-1]:
        op = x[0] & 0xffff
        if op in (OP_PHI, OP_LOOP_MERGE, OP_SWITCH, OP_FUNCTION_CALL, OP_CONTROL_BARRIER, OP_MEMORY_BARRIER, OP_KILL, OP_UNREACHABLE, OP_IMAGE_WRITE,
                  OP_EMIT_VERTEX, OP_EMIT_STREAM_VERTEX, 4416, 5380): raise _Skip   # (4416 OpTerminateInvocation, 5380 OpDemoteToHelperInvocation)
        if op == OP_LABEL: cur = x[1]; ins = []; order.append(cur); continue
        if op in (OP_BRANCH, OP_BRANCH_CONDITIONAL, OP_RETURN, OP_RETURN_VALUE):
            merge = ins.pop() if ins and ins[-1][0] & 0xffff == OP_SELECTION_MERGE else None
            if op == OP_BRANCH_CONDITIONAL and merge is None: raise _Skip
            blocks[cur] = (ins, merge, x); continue
        ins.append(x)
    if not order: raise _Skip
    variables = [x for x in blocks[order[0]][0] if x[0] & 0xffff == OP_VARIABLE]
    blocks[order[0]] = ([x for x in blocks[order[0]][0] if x[0] & 0xffff != OP_VARIABLE],) + blocks[order[0]][1:]
    # result types and pointer origins
    ty, origin = {}, {}
    for p in head: ty[p[2]] = p[1]
    for v in variables: ty[v[2]] = v[1]; origin[v[2]] = 'fn'
    for p in head:
        if m.op(p[1]) == OP_TYPE_POINTER and m.pointee(p[1])[0] == SC_FUNCTION: origin[p[2]] = 'fn'
    psb_addr, psb_chain = {}, {}
    for blk in blocks.values():
        for x in blk[0]:
            op = x[0] & 0xffff
            if op not in NO_RESULT_OPS and op not in ATOMIC_OPS | {OP_STORE} and len(x) > 2: ty[x[2]] = x[1]
            elif op in ATOMIC_OPS and op != 228: ty[x[2]] = x[1]   # (228 OpAtomicStore has no result)
            if op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN):
                base = x[3]
                origin[x[2]] = 'log' if base == log_var and log_var is not None else 'map' if base == map_var else origin.get(base, 'other')
                if origin[x[2]] == 'psb': psb_chain[x[2]] = x   # a member or element of a physical pointer's pointee
            elif op == OP_CONVERT_U_TO_PTR: origin[x[2]] = 'psb'; psb_addr[x[2]] = (x[3], x[1])
    if ret_t != m.find_type(OP_TYPE_VOID) and m.op(ret_t) not in (OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_BOOL): raise _Skip
    boolean = m.add_type(OP_TYPE_BOOL); TRUE = const_bool(m, True)
    out = []
    def em(op, t, *ops):
        r = m.new_id(); out.append([(3 + len(ops)) << 16 | op, t, r, *ops]); ty[r] = t; return r
    ands = {}
    def AND(p, q):
        if p == TRUE: return q
        if q == TRUE: return p
        k = (p, q)
        if k not in ands: ands[k] = em(OP_LOGICAL_AND, boolean, p, q)
        return ands[k]
    nots = {}
    def NOT(p):
        if p not in nots: nots[p] = em(OP_LOGICAL_NOT, boolean, p)
        return nots[p]
    # the sink, read from the table once, at the top
    mac = next(x for b in blocks.values() for x in b[0] if x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and x[3] == map_var and len(x) == 6)
    p_word, idx0, idx_t = mac[1], mac[4], m.consts[mac[5]][0] if mac[5] in m.consts else None
    if idx_t is None: idx_t = m.consts[idx0][0]
    word_t = m.pointee(p_word)[1]
    sink = {}
    def SINK(t):
        if t in sink: return sink[t]
        if 'v2' not in sink:
            lo, hi = (em(OP_LOAD, word_t, em(OP_ACCESS_CHAIN, p_word, map_var, idx0, m.add_const(idx_t, w))) for w in (sink_word, sink_word + 1))
            sink['v2'] = em(OP_COMPOSITE_CONSTRUCT, m.add_type(OP_TYPE_VECTOR, word_t, 2), lo, hi)
        v2 = sink['v2']
        if t == ty[v2]: sink[t] = v2
        elif m.op(t) == OP_TYPE_INT and m.types[t][2] == 64: sink[t] = em(OP_BITCAST, t, v2)
        else: raise _Skip
        return sink[t]
    state = {'alive': TRUE, 'ret': None}
    if ret_t != m.find_type(OP_TYPE_VOID): state['ret'] = const_bool(m, False) if m.op(ret_t) == OP_TYPE_BOOL else m.add_const(ret_t, 0)
    ffff = None
    def Q(P): return AND(P, state['alive'])
    def safe_ptr(ptr, q):   # a physical pointer that reaches the sink where q is false (an access chain made again on the sink's pointer)
        if q == TRUE: return ptr
        if ptr in psb_chain:
            x = psb_chain[ptr]
            return em(x[0] & 0xffff, x[1], safe_ptr(x[3], q), *x[4:])
        addr, pt = psb_addr[ptr]
        return em(OP_CONVERT_U_TO_PTR, pt, em(OP_SELECT, ty[addr], q, addr, SINK(ty[addr])))
    def emit(x, P):
        nonlocal ffff
        op = x[0] & 0xffff
        if op == OP_STORE:
            ptr, val = x[1], x[2]; o = origin.get(ptr); q = Q(P)
            if o == 'fn':
                if q == TRUE or ptr in one_path: out.append(x); return
                pt = m.pointee(ty[ptr])[1]
                if m.op(pt) not in (OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_BOOL, OP_TYPE_VECTOR): raise _Skip
                old = em(OP_LOAD, pt, ptr); out.append([3 << 16 | OP_STORE, ptr, em(OP_SELECT, pt, q, val, old)]); return
            if o == 'psb': out.append([x[0], safe_ptr(ptr, q), *x[2:]]); return
            if o == 'log' and m.consts.get(val, (None, [None]))[1] == [1] and cas_sem is not None:
                if q == TRUE: out.append(x); return
                t = m.pointee(ty[ptr])[1]
                em(OP_ATOMIC_UMAX, t, ptr, cas_sem[0], cas_sem[1], em(OP_SELECT, t, q, val, m.add_const(t, 0))); return
            raise _Skip
        if op in ATOMIC_OPS:
            ptr = x[3]; o = origin.get(ptr); q = Q(P)
            if o == 'psb': out.append([x[0], x[1], x[2], safe_ptr(ptr, q), *x[4:]]); return
            if o == 'log' and op == OP_ATOMIC_COMPARE_EXCHANGE and m.consts.get(x[8], (None, [None]))[1] == [0]:
                if q == TRUE: out.append(x); return
                if ffff is None: ffff = m.add_const(x[1], 0xFFFFFFFF)
                out.append([*x[:8], em(OP_SELECT, x[1], q, x[8], ffff)]); return
            raise _Skip
        if op == OP_LOAD and origin.get(x[3]) == 'psb':
            out.append([x[0], x[1], x[2], safe_ptr(x[3], Q(P)), *x[4:]]); return
        out.append(x)
    # the path of conditions each block is on: a variable read and written only on one path (the translator's temporaries, nearly all) is stored unconditionally;
    # selecting its old value would keep that value alive across the helper (Turnip: register pressure 401 → 2118 on one program)
    path, seen0 = {}, set()
    def paths(label, stop, key):
        while label != stop:
            if label in seen0 or label not in blocks: raise _Skip
            seen0.add(label); path[label] = key
            _, merge, term = blocks[label]; op = term[0] & 0xffff
            if op == OP_BRANCH: label = term[1]; continue
            if op == OP_BRANCH_CONDITIONAL:
                c, t, f, mg = term[1], term[2], term[3], merge[1]
                if t != mg: paths(t, mg, key + ((c, 1),))
                if f != mg: paths(f, mg, key + ((c, 0),))
                label = mg; continue
            return
    paths(order[0], None, ())
    var_paths = {}
    for label, (ins, _, _) in blocks.items():
        for x in ins:
            op = x[0] & 0xffff
            v = x[1] if op == OP_STORE else x[3] if op == OP_LOAD else None
            if v is not None and origin.get(v) == 'fn': var_paths.setdefault(v, set()).add(path.get(label))
    one_path = {v for v, ps in var_paths.items() if len(ps) == 1 and v not in {p[2] for p in head}}
    cas = next((x for b in blocks.values() for x in b[0] if x[0] & 0xffff == OP_ATOMIC_COMPARE_EXCHANGE), None)
    cas_sem = (cas[4], cas[5]) if cas is not None else None
    seen = set()
    def walk(label, stop, P):
        while label != stop:
            if label in seen or label not in blocks: raise _Skip
            seen.add(label)
            ins, merge, term = blocks[label]
            for x in ins: emit(x, P)
            op = term[0] & 0xffff
            if op == OP_BRANCH: label = term[1]; continue
            if op == OP_BRANCH_CONDITIONAL:
                c, t, f, mg = term[1], term[2], term[3], merge[1]
                if t != mg: walk(t, mg, AND(P, c))
                if f != mg: walk(f, mg, AND(P, NOT(c)))
                label = mg; continue
            q = Q(P)   # a return
            if op == OP_RETURN_VALUE: state['ret'] = term[1] if q == TRUE else em(OP_SELECT, ret_t, q, term[1], state['ret'])
            state['alive'] = AND(state['alive'], NOT(P)) if P != TRUE else const_bool(m, False)
            return
        return
    walk(order[0], None, TRUE)
    if set(order) - seen: raise _Skip   # a block no path reached
    end = [2 << 16 | OP_RETURN_VALUE, state['ret']] if state['ret'] is not None else [1 << 16 | OP_RETURN]
    return [fn, *head, [2 << 16 | OP_LABEL, order[0]], *variables, *out, end, body[-1]]


# ---- the raw global memory helpers' pool search as a loop (gmem_loop)
OP_LOOP_MERGE_, OP_PHI_, OP_BRANCH_, OP_BRANCH_COND_, OP_LABEL_, OP_SELECT_, OP_IADD_, OP_SHR_LOGICAL_, OP_IEQUAL_ = 246, 245, 249, 250, 248, 169, 128, 194, 170
OP_ULESS_THAN = 176
LOOP_CONTROL_DONT_UNROLL = 2
_TERMINATORS = {249, 250, 251, 252, 253, 254, 255, 4416}


def gmem_loop(m, map_var):
    """The pool search of the translator's raw global memory helpers (GlobalToStorage.TryGenerateRawStorageOp: the largest table entry whose start is at or below
    the address, found with 12 halving steps 2048, 1024 .. 1, each two loads of the map table, three compares and a select) made one loop that NIR does not unroll
    (DontUnroll), instead of 12 unrolled steps. Turnip inlines every call: Ocarina of Time's shaders with 100 to 253 such loads in one block (gmem_flatten) become
    80,000 ir3 instructions in which the scheduler overlaps all the searches, past the register file: 5.5 KiB of private memory a fibre (spills), so Turnip's device-wide
    private memory BOs (per wave and per fibre, sized by the largest program and never shrunk, tu_setup_pvtmem) reach 8 KiB x 4096 fibres x 6 SPs = 192 MiB each, KGSL
    memory that is never swapped. With the loop each search runs alone: no spills in those shaders (and 8 waves instead of 4 in most), 4x fewer instructions, ~10x faster
    pipeline compiles. The searches no longer overlap each other (more waves hide the latency instead).
    Only a search whose 12 steps are the same instructions (step k: the step constant and the previous step's index, every other operand the step's own values or values
    from before the search) and whose intermediate values nothing after it reads is taken; anything else stays as it is.
    map_var: the table's variable (vk_android_build.py storage_slots: KIND_GMEM_MAP). → number of searches made loops"""
    if map_var is None: return 0
    entry = m.entry_function()
    done = 0
    for f in [x[2] for x in m.ins if x[0] & 0xffff == OP_FUNCTION and x[2] != entry]:
        while True:   # (a helper with more than one search: each in turn)
            a, e = m.function_range(f)
            new = _loop_search(m, m.ins[a:e + 1], map_var)
            if new is None: break
            a, e = m.function_range(f)   # (a type or constant the rewrite added went in before the functions)
            m.ins[a:e + 1] = new
            done += 1
    if done: m.index()
    return done


GMEM_LOOP_STAGES = frozenset({'cs'})   # where gmem_loop applies: compute. In Ocarina of Time's 683x384 opaque pass its vertex shaders' loops (802 vs, 605 ps changed in
# v-4eb4a4fafd13) came with a pass 8x slower on the phone (33-48 ms → ~370 ms), while compute gained (9e6b00b0: 20 → 9.1 ms); android/SHADERS.md#build-speed
GMEM_CACHE_WORDS = 6   # the words of an entry gmem_cache keeps: start, end, delta (low, high each; dmsh_abi.h DMSH_GMEM_MAP_ENTRY). On the A830 (offline ir3,
# b1f06157308e3b22) 6 runs the fewest instructions, spills and loads where the cache hits; 0 (the index only) the fewest in all (gmc/IMPL.md)
GMEM_CACHE = False   # the other stages' searches behind the invocation's cached entry (gmem_cache); False: left unrolled, as in v-a25cc83eed63. Off: with it on
# (v-0da35c39552a) Ocarina of Time measured slower on the phone (Turnip, fb11: +14 % per vertex invocation in the house, +10-40 % at other spots; session-2026-11-07/gmc/FB11.md)


def gmem_cache(m, map_var, words=None):
    """The pool searches of a graphics stage's raw global memory helpers (gmem_loop's shape) behind the entry the invocation's last search found. Unrolled, each search is
    12 dependent steps of 2 table loads: Ocarina of Time's forest vertex shaders call one helper 253 times (a matrix palette read through one pointer), and Turnip inlines
    them into 82,257 ir3 instructions with 1,236 spill pairs, 2.5-3.5 us per vertex (lights/NEXT.md); as plain loops (gmem_loop) the 253 x 12 steps ran one after another,
    8x slower still. Here Private variables (per invocation) hold the entry g the last search found, its first `words` table words (start, end, delta: 6) and the start of
    entry g + 1 (~0 past the table's last entry), initialised to entry 0 with all words 0: an empty range, so the first search runs. Before each search: if g's start is at
    or below the address and g + 1's is not, the search's answer is g (its answer is the last entry whose start is at or below the address; the table is sorted by start:
    dmsh_abi.h DMSH_GMEM_MAP_ENTRIES, vka_draw.c gmem_map, the search's own assumption), so g is taken; else the search runs as gmem_loop's loop (DontUnroll) and its answer
    and words go into the cache. The two conditions are the search's own step instructions with the start words taken from the cache, and the words the helper reads of the
    answer after the search (end, delta) come from the cache as well; the pool check, the sink select and the access stay as they are. So the answer and every word are
    the search's in every case (exact: entry 0, the sink, the end of a pool and the last entry included). words 0: only g is kept and the starts of g and g + 1 (g at most
    4094) are read from the table at each search (fewer registers, more loads). Helper lanes, demoted lanes and non-uniform control flow change nothing (no derivative,
    no subgroup operation; the cache is per invocation; Private, not Function storage, as the helper is its own function and NIR inlines it).
    → number of searches given the cache"""
    if map_var is None: return 0
    words = GMEM_CACHE_WORDS if words is None else words
    entry = m.entry_function()
    var, done = None, 0
    for f in [x[2] for x in m.ins if x[0] & 0xffff == OP_FUNCTION and x[2] != entry]:
        while True:
            a, e = m.function_range(f)
            body = m.ins[a:e + 1]
            mt = _match_search(m, body, map_var)
            if mt is None or _cache_plan(m, body, mt, map_var) is None: break
            if var is None:   # the cache: Private variables, the entry's index (the search's integer type) and the start words of it and the next (the table's word type)
                i32, zero = mt[5], mt[4]
                word_t = next(x[1] for x in mt[1] if x[0] & 0xffff == OP_LOAD and x[3] in {y[2] for y in mt[1] if y[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN)})
                w0 = m.add_const(word_t, 0)
                var = []
                for name, t, init in [('dm_gmem_entry', i32, zero)] + ([(f'dm_gmem_w{k}', word_t, w0) for k in range(words)] + [('dm_gmem_next_l', word_t, w0), ('dm_gmem_next_h', word_t, w0)] if words else []):
                    pt = m.add_type(OP_TYPE_POINTER, SC_PRIVATE, t)
                    v = m.new_id(); m._insert_decl([5 << 16 | OP_VARIABLE, pt, v, SC_PRIVATE, init]); var.append(v)
                    nm = lit_string(name)
                    m.ins.insert(next(k for k, y in enumerate(m.ins) if y[0] & 0xffff in (OP_NAME, OP_DECORATE)), [(2 + len(nm)) << 16 | OP_NAME, v, *nm])
                if m.header[1] >= 0x10400:   # SPIR-V 1.4 on: every global variable the entry point uses is in its interface (the translator's modules are 1.3)
                    k = next(k for k, y in enumerate(m.ins) if y[0] & 0xffff == OP_ENTRY_POINT); m.ins[k] = m.ins[k] + var; m.ins[k][0] = len(m.ins[k]) << 16 | OP_ENTRY_POINT
                a, e = m.function_range(f); body = m.ins[a:e + 1]
            new = _loop_search(m, body, map_var, var, words)
            if new is None: break
            a, e = m.function_range(f)
            m.ins[a:e + 1] = new
            done += 1
    if done: m.index()
    return done


def gmem_loop_stage(m, map_var, stage, cache=None, pidx=None):
    """gmem_loop where it applies (GMEM_LOOP_STAGES); in another stage the page index (gmem_pidx; pidx: GMEM_PIDX when None) or gmem_cache (cache: GMEM_CACHE when
    None), or with both off the module is left as it is and the searches gmem_loop would take are counted on a copy. → (searches made loops, searches given the cache
    or the page index, searches left as they are)"""
    if stage in GMEM_LOOP_STAGES: return gmem_loop(m, map_var), 0, 0
    if map_var is None: return 0, 0, 0
    if GMEM_PIDX if pidx is None else pidx: n = gmem_pidx(m, map_var); return 0, n, gmem_loop(Module(m.data()), map_var)   # (left: the searches it refused)
    if GMEM_CACHE if cache is None else cache: return 0, gmem_cache(m, map_var), 0
    return 0, 0, gmem_loop(Module(m.data()), map_var)


GMEM_PIDX = True   # the other stages' searches replaced by the page index (gmem_pidx; offline A830 b1f06157308e3b22: 82,257 → 9,888 instructions, session-2026-11-07/pidx/GATE.md).
# The build marks the entry's gmem_map binding DMSH_VK_BF_PIDX, and the host then binds the table with the index after it (vka_draw.c gmem_map); False: v-a25cc83eed63's searches
GMEM_PIDX_WORD = 4096 * 8   # the index's header in the table's buffer, right after the table (dmsh_abi_vk.h DMSH_VK_GMEM_PIDX_WORD; vk_android_build.py checks they agree)
GMEM_PIDX_SHIFT = 12   # log2 of the index's page: 4 KiB, the placed segments' alignment (vka_mem.c pg_align: at least 4096; DMSH_VK_GMEM_PIDX_SHIFT)
_PIDX_CMP = {170, 171, 172, 174, 176, 178}   # OpIEqual, OpINotEqual, OpUGreaterThan(Equal), OpULessThan(Equal): the search step's start-against-address compares


def gmem_pidx(m, map_var):
    """The pool searches of the translator's raw global memory helpers (gmem_loop's shape: 12 unrolled steps, 24 dependent table loads) and the helper's use of the
    answer (its end: the pool check; its delta: the device address) replaced by one load from a page index: straight-line code, no branch, no loop (gmem_cache's
    per-call branch and miss loop measured slower on the phone, gmc/FB11.md). The host puts after the table (word GMEM_PIDX_WORD) the index's first guest page
    (address >> 12; the window lies below 0xfff00000 pages, 16 TiB), its page count and 2 words of 0, then for each 4 KiB page of the window and one past it an 8-byte
    record: for a page inside a table entry the device address of the page (the page's guest address + the entry's delta) with bit 0 set, else 0 (the record past the
    window is 0). Placed segments are aligned to 4 KiB (vka_mem.c pg_align) and the entries are unions of them (vka_mem_chunks), so a page lies in one entry or in none;
    a delta is a multiple of 4 KiB (device and host map the same pages), so the page's device address has its low 12 bits free.
    Exact: an address in entry e is in a page whose record is set, and the device address the record gives, (page's device address) | (address & 0xfff), is address +
    delta_e, what the helper computes from the search's answer e; its pool check (address < end_e) holds. An address in no entry: the search's answer ends at or below it
    (or is entry 0), the pool check fails; the page's record is 0 (or the window's sentinel), so the check is false here too. Where the check is false the helper uses
    the sink (entry 0's words, read as before) and its value 0, unchanged. The pool check's value is replaced by the record's bit, the device address by the record's;
    the rest of the helper (the sink select, the access, the write log) stays. Only a helper whose use of the answer is the translator's (its end words feed one boolean,
    its delta words one 64-bit address, each from the address words and those table words alone) is taken; anything else stays as it is. → searches replaced"""
    if map_var is None: return 0
    entry = m.entry_function()
    done = 0
    for f in [x[2] for x in m.ins if x[0] & 0xffff == OP_FUNCTION and x[2] != entry]:
        while True:
            a, e = m.function_range(f)
            new = _pidx_search(m, m.ins[a:e + 1], map_var)
            if new is None: break
            a, e = m.function_range(f)   # (a type or constant the rewrite added went in before the functions)
            m.ins[a:e + 1] = new
            done += 1
    if done: m.index()
    return done


def _operands(x):   # positions of the ids an instruction reads
    ps = id_positions(x)
    return [p for p in ps if p >= 3] if _result_id(x) is not None and len(x) > 2 and _result_id(x) == x[2] else [p for p in ps if x[0] & 0xffff != OP_LABEL_]


_PIDX_CASES = [(a, b) for a in (0, 1, 0xfff, 0x1000, 0xffffffff, 0x100000000, 0x41a1de8680, 0x41a1de867f, 0x41a1de8681, 0x4100000000, 0x41ffffffff, (1 << 64) - 1, 0x10ffffffff0)
               for b in (0, 1, 0xffffffff, 0x100000000, 0x41a1de8680, 0x41a1de8681, 0x41a1de867f, 0x41ffffffff, 0x4200000000, (1 << 64) - 1, (1 << 64) - 0x41a1de8680, 0x10ffffffff0)]


def _pidx_semantics(m, pdefs, word_of, addr, proot, cond, s_part, dev, t_part):
    """gmem_pidx's check that the helper's pool check is address < end and its device address is address + delta (64-bit, unsigned, wrapping), evaluated on the
    SPIR-V itself over edge cases (carries, equal high words, the ends of the range): the replacement is exact only for that formula"""
    M32 = (1 << 32) - 1
    def run(part, top, words, a):
        val = {}
        def get(v):
            if v in val: return val[v]
            if v in m.consts: return m.consts[v][1][0] if len(m.consts[v][1]) == 1 else m.consts[v][1][0] | m.consts[v][1][1] << 32
            q = proot(v)
            if q == addr[0]: return a & M32
            if q == addr[1]: return a >> 32
            raise KeyError(v)
        for v in sorted(part | set(words), key=lambda v: order[v]):
            x = pdefs[v]; op = x[0] & 0xffff; o = [get(x[p]) for p in range(3, len(x))] if op not in (OP_LOAD,) else []
            t = x[1]; w = m.int_type(t)
            if v in words: r = words[v]
            elif op == OP_LOAD: r = get(v) if proot(v) in (addr[0], addr[1]) else None
            elif op in (OP_BITCAST, OP_COPY_OBJECT, OP_CONVERT_U_TO_PTR):
                r = o[0]
                if w == (64, 0) or w == (64, 1): r = r[0] | r[1] << 32 if isinstance(r, tuple) else r
                elif m.op(t) == OP_TYPE_VECTOR and not isinstance(r, tuple): r = (r & M32, r >> 32)
            elif op == OP_ULESS_THAN: r = o[0] < o[1]
            elif op == 178: r = o[0] <= o[1]
            elif op == OP_IEQUAL_: r = o[0] == o[1]
            elif op == OP_LOGICAL_AND: r = o[0] and o[1]
            elif op == OP_LOGICAL_OR: r = o[0] or o[1]
            elif op == OP_LOGICAL_NOT: r = not o[0]
            elif op == 154: r = any(o[0])   # OpAny
            elif op == 155: r = all(o[0])   # OpAll
            elif op == OP_COMPOSITE_CONSTRUCT: r = tuple(o)
            elif op == OP_IADD_: r = (o[0] + o[1]) & ((1 << (w[0] if w else 32)) - 1)
            elif op == OP_SELECT_: r = o[1] if o[0] else o[2]
            else: return None
            if r is None: return None
            val[v] = r
        r = val[top]
        return r[0] | r[1] << 32 if isinstance(r, tuple) else r
    order = {v: i for i, v in enumerate(pdefs)}
    end_w = {v: k for v, k in word_of.items() if k in (2, 3)}; del_w = {v: k for v, k in word_of.items() if k in (4, 5)}
    try:
        for a, b in _PIDX_CASES:
            if run(s_part, cond, {v: (b if k == 2 else b >> 32) & M32 for v, k in end_w.items()}, a) != (a < b): return False
            if run(t_part, dev, {v: (b if k == 4 else b >> 32) & M32 for v, k in del_w.items()}, a) != (a + b) & ((1 << 64) - 1): return False
    except (KeyError, TypeError, IndexError):
        return False
    return True


def _pidx_search(m, body, map_var):
    """gmem_pidx's rewrite of the first search in body (None: none it takes). The address words are what the step compares with the candidate's start words (word 0
    with the low, word 1 with the high word): values from before the search, or loads of a pointer the step reads (the helper's parameters), loaded again here"""
    mt = _match_search(m, body, map_var)
    if mt is None: return None
    steps, tmpl, pos_prev, pos_step, zero, i32, last = mt
    s0, final = steps[0][0], body[last][2]
    types = {x[2]: x[1] for x in body if len(x) > 2 and _result_id(x) == x[2] and x[0] & 0xffff != OP_FUNCTION}
    chain_t = next(x for x in tmpl if x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and x[3] == map_var and len(x) == 6)
    word_t = next(x[1] for x in tmpl if x[0] & 0xffff == OP_LOAD and x[3] == chain_t[2])
    post = [list(x) for x in body[last + 1:]]

    def analyse(code, base):   # → (root of a value, the table word a chain into the map reads: base * 8 + k)
        defs = {_result_id(x): x for x in code}; defs.pop(None, None)
        def x8(v):
            x = defs.get(v)
            if x is None or len(x) != 5: return False
            c = (m.consts.get(x[4]) or (0, [0]))[1]
            return (base is None or x[3] == base) and (x[0] & 0xffff == OP_IMUL and c == [8] or x[0] & 0xffff == OP_SHL_LOGICAL and c == [3])
        def word_k(chain):
            if chain[0] & 0xffff not in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) or chain[3] != map_var or len(chain) != 6: return None
            if x8(chain[5]): return 0
            ad = defs.get(chain[5])
            if ad is None or ad[0] & 0xffff != OP_IADD_ or len(ad) != 5: return None
            for u, v in ((ad[3], ad[4]), (ad[4], ad[3])):
                k = m.consts.get(v)
                if k is not None and len(k[1]) == 1 and x8(u): return k[1][0]
            return None
        def root(v):   # → ('table', k), ('load', pointer, type), ('id', id), ('const', id) or None
            while v in defs and defs[v][0] & 0xffff in (OP_BITCAST, OP_COPY_OBJECT): v = defs[v][3]
            if v in m.consts: return ('const', v)
            x = defs.get(v)
            if x is None: return ('id', v) if v in types else None
            if x[0] & 0xffff != OP_LOAD: return None
            if x[3] in defs:
                k = word_k(defs[x[3]])
                return None if k is None else ('table', k)
            return ('load', x[3], x[1])
        return defs, word_k, root

    # the address words, from the search's compares
    defs, word_k, root = analyse(tmpl, None)
    addr = {}
    for x in tmpl:
        if x[0] & 0xffff not in _PIDX_CMP or len(x) != 5: continue
        r = [root(x[3]), root(x[4])]
        t = [q for q in r if q and q[0] == 'table']; o = [q for q in r if q and q[0] in ('load', 'id')]
        if len(t) != 1: continue
        if len(o) != 1 or t[0][1] not in (0, 1) or addr.setdefault(t[0][1], o[0]) != o[0]: return None
    if set(addr) != {0, 1}: return None

    # after the search: the answer's words 2, 3 (end) and 4, 5 (delta), and what depends on them
    pdefs, pword_k, proot = analyse(post, final)
    word_of = {}
    for x in post:
        if x[0] & 0xffff == OP_LOAD and x[3] in pdefs:
            k = pword_k(pdefs[x[3]])
            if k is not None: word_of[x[2]] = k
    if sorted(set(word_of.values())) != [2, 3, 4, 5]: return None
    dep, users = {}, {}
    for x in post:
        r = _result_id(x)
        d = {word_of[r]} if r in word_of else set()
        for p in _operands(x):
            d |= dep.get(x[p], set()); users.setdefault(x[p], []).append(r)
        if r is not None: dep[r] = d
    S = {v for v, d in dep.items() if d and d <= {2, 3}}; T = {v for v, d in dep.items() if d and d <= {4, 5}}
    def frontier(group, allowed):   # the one value of the group everything else of it feeds (the rest reads only it: its negation, say), its inputs the address words
        # or constants → (that value, the group's part that computes it), or None
        ops = {v: {pdefs[v][p] for p in _operands(pdefs[v])} for v in group}
        def ancestors(c):
            anc, todo = set(), [c]
            while todo:
                for u in ops[todo.pop()] & group:
                    if u not in anc: anc.add(u); todo.append(u)
            return anc
        for c, anc in sorted(((c, ancestors(c)) for c in group), key=lambda t: (-len(t[1]), t[0])):
            part = anc | {c}
            if not all(ops[v] & group <= (group - part) | {c} for v in group - part): continue
            if any(any(u is None or u not in part for u in users.get(v, ())) for v in anc): continue
            for v in part:
                x = pdefs[v]
                if x[0] & 0xffff not in allowed: return None
                for p in _operands(x):
                    u = x[p]
                    if u in part or u in word_of: continue
                    q = proot(u)
                    if q is None or q[0] == 'table' or (q[0] != 'const' and q not in addr.values()): return None
            return c, part
        return None
    loads = {v for v in word_of}
    fc = frontier(S - loads, {OP_BITCAST, OP_COPY_OBJECT, OP_LOAD, OP_ULESS_THAN, 178, OP_IEQUAL_, OP_LOGICAL_AND, OP_LOGICAL_OR, OP_LOGICAL_NOT, 154, 155, OP_COMPOSITE_CONSTRUCT})
    fd = frontier(T - loads, {OP_BITCAST, OP_COPY_OBJECT, OP_LOAD, OP_IADD_, OP_ULESS_THAN, OP_SELECT_, OP_COMPOSITE_CONSTRUCT, OP_CONVERT_U_TO_PTR})
    if fc is None or fd is None: return None
    (cond, s_part), (dev, t_part) = fc, fd
    if not _pidx_semantics(m, pdefs, word_of, addr, proot, cond, s_part, dev, t_part): return None
    boolean = m.add_type(OP_TYPE_BOOL); v2 = m.add_type(OP_TYPE_VECTOR, word_t, 2)
    if types.get(cond) != boolean: return None
    dev_t = types.get(dev)
    if dev_t != v2 and m.int_type(dev_t) != (64, 0): return None

    out = []
    nid = m.new_id
    def em(op, t, *ops): r = nid(); out.append([(3 + len(ops)) << 16 | op, t, r, *ops]); return r
    def wc(v): return m.add_const(word_t, v)
    def as_word(q):
        v = em(OP_LOAD, q[2], q[1]) if q[0] == 'load' else q[1]
        return v if (q[2] if q[0] == 'load' else types[v]) == word_t else em(OP_BITCAST, word_t, v)
    def table_word(w):
        return em(OP_LOAD, word_t, em(chain_t[0] & 0xffff, chain_t[1], map_var, chain_t[4], em(OP_BITCAST, i32, w)))
    glsl = ensure_glsl(m)
    lo, hi = as_word(addr[0]), as_word(addr[1])
    base, pages = table_word(wc(GMEM_PIDX_WORD)), table_word(wc(GMEM_PIDX_WORD + 1))
    page = em(OP_BITWISE_OR, word_t, em(OP_SHL_LOGICAL, word_t, em(OP_EXT_INST, word_t, glsl, GLSL_UMIN, hi, wc((1 << GMEM_PIDX_SHIFT) - 1)), wc(32 - GMEM_PIDX_SHIFT)),
              em(OP_SHR_LOGICAL_, word_t, lo, wc(GMEM_PIDX_SHIFT)))   # (the high word clamped: an address at or past 2^44 lands past the window)
    pg = em(OP_EXT_INST, word_t, glsl, GLSL_UMIN, em(OP_ISUB, word_t, page, base), pages)   # below the window it wraps past it; past it: the sentinel record
    rec = em(OP_IADD_, word_t, em(OP_SHL_LOGICAL, word_t, pg, wc(1)), wc(GMEM_PIDX_WORD + 4))
    p_lo, p_hi = table_word(rec), table_word(em(OP_IADD_, word_t, rec, wc(1)))
    valid = em(OP_INOT_EQUAL, boolean, em(OP_BITWISE_AND, word_t, p_lo, wc(1)), wc(0))
    mask = wc(((1 << 32) - 1) ^ ((1 << GMEM_PIDX_SHIFT) - 1))
    d_lo = em(OP_BITWISE_OR, word_t, em(OP_BITWISE_AND, word_t, p_lo, mask), em(OP_BITWISE_AND, word_t, lo, wc((1 << GMEM_PIDX_SHIFT) - 1)))
    d = em(OP_COMPOSITE_CONSTRUCT, v2, d_lo, p_hi)
    if dev_t != v2: d = em(OP_BITCAST, dev_t, d)
    gone = s_part | t_part | loads
    fin = {final}   # the answer and what is computed from it alone (its words' indices and chains): unread once the words are gone
    for x in post:
        r = _result_id(x)
        if r is not None and x[0] & 0xffff in (OP_IMUL, OP_SHL_LOGICAL, OP_IADD_, OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and any(x[p] in fin for p in _operands(x)): fin.add(r)
    keep = []
    for x in post:
        r = _result_id(x)
        if r in gone or r in fin: continue
        for p in _operands(x):
            if x[p] == cond: x[p] = valid
            elif x[p] == dev: x[p] = d
            elif x[p] in fin or x[p] in gone: return None   # (the answer or its words read elsewhere: not the translator's helper)
        keep.append(x)
    return body[:s0] + out + keep


def _result_id(x):   # the id an instruction defines (types and constants do not occur in a function)
    op = x[0] & 0xffff
    if op == OP_LABEL_: return x[1]
    if op in NO_RESULT_OPS or op in (OP_BRANCH_, OP_BRANCH_COND_, OP_LOOP_MERGE_, OP_SELECTION_MERGE, 251, 252, 255): return None
    return x[2] if len(x) > 2 else None


def _cache_plan(m, body, mt, map_var):
    """What gmem_cache needs of a search, or None when it cannot take it: the step's condition must depend on the candidate entry only through the loads of its start
    (table words cand * 8 + 0 and + 1, each one access chain and one load) and on nothing else of the step's operands (the previous index, the step), and the search's
    block must be no loop header (its merge instruction would move to another block). → (condition id, {load id: word}, access chain template, the instructions the
    condition needs, in order)"""
    steps, tmpl, pos_prev, pos_step, last = mt[0], mt[1], mt[2], mt[3], mt[6]
    for x in body[last + 1:]:
        if x[0] & 0xffff == OP_LOOP_MERGE_: return None
        if x[0] & 0xffff in _TERMINATORS: break
    defs = {_result_id(x): (i, x) for i, x in enumerate(tmpl)} ; defs.pop(None, None)
    cand = tmpl[0][2]   # (the step's first instruction: cand = IAdd(prev, step))
    def entry_x8(v):   # v = cand * 8 (IMul by 8 or ShiftLeftLogical by 3)
        mu = defs.get(v, (0, None))[1]
        if mu is None or len(mu) != 5: return False
        if mu[0] & 0xffff == OP_IMUL: return cand in mu[3:5] and m.consts.get(mu[4] if mu[3] == cand else mu[3]) == (mu[1], [8])
        return mu[0] & 0xffff == OP_SHL_LOGICAL and mu[3] == cand and (m.consts.get(mu[4]) or (0, [0]))[1] == [3]
    def word_of(chain):   # the start word an access chain into the table reads: index cand * 8 + k, k 0 or 1 (k 0: the product itself or IAdd with 0)
        x = defs.get(chain, (0, None))[1]
        if x is None or x[0] & 0xffff not in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) or x[3] != map_var or len(x) != 6: return None
        if entry_x8(x[5]): return 0, x
        ad = defs.get(x[5], (0, None))[1]
        if ad is None or ad[0] & 0xffff != OP_IADD_ or len(ad) != 5: return None
        for u, v in ((ad[3], ad[4]), (ad[4], ad[3])):
            k = m.consts.get(v)
            if k is not None and k[1] in ([0], [1]) and entry_x8(u): return k[1][0], x
        return None
    cond = body[steps[0][1]][3]
    if cond not in defs: return None
    loads, need, chain_t, todo = {}, set(), None, [cond]
    while todo:
        v = todo.pop()
        if v in need or v in loads: continue
        i, x = defs[v]
        if x[0] & 0xffff == OP_LOAD and x[3] in defs:
            w = word_of(x[3])
            if w is None: return None
            loads[v] = w[0]; chain_t = w[1]; continue
        if x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN, OP_STORE) or v == cand: return None
        if any((i, j) in pos_prev or (i, j) in pos_step for j in range(3, len(x))): return None
        need.add(v)
        todo += [x[j] for j in id_positions(x) if j >= 3 and x[j] in defs]
    if sorted(loads.values()) != [0, 1]: return None
    return cond, loads, chain_t, [x for x in tmpl if _result_id(x) in need]


def _match_search(m, body, map_var):
    """The 12 unrolled steps of a pool search in a helper's body (gmem_loop's rules) → (steps, template, prev positions, step positions, zero, i32, last select), or None"""
    if not any(x[0] & 0xffff == OP_CONVERT_U_TO_PTR for x in body): return None
    cint = lambda c: m.consts[c][1][0] if c in m.consts and m.int_type(m.consts[c][0]) and m.int_type(m.consts[c][0])[0] == 32 else None
    # the first step: cand = IAdd(i32, 0, 2048) (either order), its select = Select(i32, cond, cand, 0)
    k0 = None
    for k, x in enumerate(body):
        if x[0] & 0xffff == OP_IADD_ and len(x) == 5 and {cint(x[3]), cint(x[4])} == {0, 2048} and x[3] != x[4]: k0 = k; break
    if k0 is None: return None
    i32 = body[k0][1]; zero = body[k0][3] if cint(body[k0][3]) == 0 else body[k0][4]
    steps = []   # (start index, select index, prev id, step const id)
    prev, start, step = zero, k0, 2048
    while True:
        x = body[start]
        if not (x[0] & 0xffff == OP_IADD_ and x[1] == i32 and len(x) == 5 and set(x[3:5]) == {prev, _cid(m, i32, step, x)}): return None
        cand = x[2]
        sel = next((k for k in range(start + 1, len(body)) if body[k][0] & 0xffff == OP_SELECT_ and body[k][1] == i32 and body[k][4] == cand and body[k][5] == prev), None)
        if sel is None: return None
        steps.append((start, sel, prev, _cid(m, i32, step, x)))
        prev = body[sel][2]
        if step == 1: break
        step //= 2
        start = sel + 1
        while start < len(body) and not (body[start][0] & 0xffff == OP_IADD_ and len(body[start]) == 5 and prev in body[start][3:5]): start += 1
        if start == len(body) or start - steps[-1][1] > 1: return None   # (the steps follow each other)
    if len(steps) != 12: return None
    s0, e0, _, c0 = steps[0]
    tmpl = body[s0:e0 + 1]
    if not any(x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and x[3] == map_var for x in tmpl): return None
    if any(x[0] & 0xffff in (OP_STORE, OP_FUNCTION_CALL, OP_LABEL) or (x[0] & 0xffff) in _TERMINATORS for x in tmpl): return None
    own0 = {_result_id(x) for x in tmpl} - {None}
    # every step the same instructions as step 0: at the operands where step 1 has its previous index and its step constant (positions), step k has its own;
    # an operand that is a value of step 0 is the same value of step k; every other operand (types, constants, values from before the search) the same
    pos_prev, pos_step = set(), set()
    s1, e1, p1, c1 = steps[1]
    for i, (x0, x1) in enumerate(zip(tmpl, body[s1:e1 + 1])):
        if len(x0) != len(x1): return None
        for j in range(3, len(x0)):
            if x0[j] == zero and x1[j] == p1: pos_prev.add((i, j))
            elif x0[j] == c0 and x1[j] == c1: pos_step.add((i, j))
    if not pos_prev or not pos_step: return None
    for (s, e_, prev_k, ck) in steps[1:]:
        seg = body[s:e_ + 1]
        if len(seg) != len(tmpl): return None
        ren = {}
        for i, (x0, xk) in enumerate(zip(tmpl, seg)):
            if len(x0) != len(xk) or x0[0] != xk[0] or x0[1] != xk[1]: return None
            for j in range(2, len(x0)):
                u, v = x0[j], xk[j]
                if j == 2: ren[u] = v; continue
                want = prev_k if (i, j) in pos_prev else ck if (i, j) in pos_step else ren[u] if u in own0 else u
                if v != want: return None
    # nothing after the search reads a value of a step but the last select (and nothing in step 0 is read from outside the search but by its own step)
    last = steps[-1][1]; inner = set()
    for (s, e_, _, _) in steps: inner |= {_result_id(x) for x in body[s:e_ + 1]} - {None}
    final = body[last][2]; inner.discard(final)
    for k, x in enumerate(body):
        if steps[0][0] <= k <= last: continue
        if any(v in inner for v in x[1:]): return None
    return steps, tmpl, pos_prev, pos_step, zero, i32, last


def _loop_search(m, body, map_var, cache_var=None, words=0):
    """gmem_loop's rewrite of the first search in body (None: none it takes). cache_var (gmem_cache): its Private variables (entry, `words` words, the next start);
    where the search would find the cached entry again, the loop is skipped (see gmem_cache)"""
    mt = _match_search(m, body, map_var)
    if mt is None: return None
    steps, tmpl, pos_prev, pos_step, zero, i32, last = mt
    s0, e0, _, c0 = steps[0]
    final = body[last][2]
    # the block that holds the search: split it, the loop between
    lab = max(k for k in range(s0) if body[k][0] & 0xffff == OP_LABEL)
    lab_id = body[lab][1]
    plan = None
    if cache_var is not None:
        plan = _cache_plan(m, body, mt, map_var)
        if plan is None: return None
    nid = m.new_id
    H, B, C, M = nid(), nid(), nid(), nid()
    idx, st, stn, z = nid(), nid(), nid(), nid()
    boolean = m.add_type(OP_TYPE_BOOL); one = m.add_const(i32, 1)

    def instance(prev, step):   # the step's instructions over these operands, fresh ids → (instructions, renaming)
        fresh = {_result_id(x): nid() for x in tmpl}
        ins = []
        for i, x in enumerate(tmpl):
            y = list(x); y[2] = fresh[x[2]]
            for j in range(3, len(y)):
                y[j] = prev if (i, j) in pos_prev else step if (i, j) in pos_step else fresh.get(y[j], y[j])
            ins.append(y)
        return ins, fresh
    loop_body, fresh = instance(idx, st)
    idxn = fresh[body[e0][2]]
    pre = body[:s0]
    post = [list(x) for x in body[last + 1:]]
    head = lab_id   # the loop's entry edge comes from here
    check = []
    if cache_var is not None:
        P, J = nid(), nid()
        cond, loads, chain_t, need = plan
        word_t = next(x[1] for x in tmpl if _result_id(x) in loads)
        v_entry, v_w, v_hi = cache_var[0], cache_var[1:1 + words], cache_var[1 + words:]
        c8 = m.add_const(i32, 8)
        def em(op, t, *ops, to=None):
            r = nid(); (check if to is None else to).append([(3 + len(ops)) << 16 | op, t, r, *ops]); return r
        def table_word(e_, k, to=None):   # word k of entry e_, loaded as the search loads it
            return em(OP_LOAD, word_t, em(chain_t[0] & 0xffff, chain_t[1], map_var, chain_t[4], em(OP_IADD_, i32, em(OP_IMUL, i32, e_, c8, to=to), m.add_const(i32, k), to=to), to=to), to=to)
        def cond_on(word):   # the step's condition with its start words from word(k) → its id
            ren = {r: word(loads[r]) for r in (_result_id(x) for x in tmpl) if r in loads}
            for x in need:
                y = list(x)
                for j in id_positions(x):
                    if j >= 3: y[j] = ren.get(y[j], y[j])
                ren[x[2]] = y[2] = nid()
                check.append(y)
            return ren[cond]
        g = em(OP_LOAD, i32, v_entry)
        if words:   # the start words of entry g and g + 1 kept in the cache
            gc = g
            ca = cond_on(lambda k: em(OP_LOAD, word_t, v_w[k])); cb = cond_on(lambda k: em(OP_LOAD, word_t, v_hi[k]))
        else:   # read from the table: entry g and g + 1 (g at most 4094)
            gc = em(OP_SELECT_, i32, em(OP_ULESS_THAN, boolean, g, m.add_const(i32, 4095)), g, m.add_const(i32, 4094))
            g1 = em(OP_IADD_, i32, gc, one)
            ca = cond_on(lambda k: table_word(gc, k)); cb = cond_on(lambda k: table_word(g1, k))
        hit = em(OP_LOGICAL_AND, boolean, ca, em(OP_LOGICAL_NOT, boolean, cb))   # entry g starts at or below the address, entry g + 1 does not
        check += [[3 << 16 | OP_SELECTION_MERGE, J, 0], [4 << 16 | OP_BRANCH_COND_, hit, J, P], [2 << 16 | OP_LABEL_, P]]
        head = P
    out = pre + check + [[2 << 16 | OP_BRANCH_, H],
                 [2 << 16 | OP_LABEL_, H], [7 << 16 | OP_PHI_, i32, idx, zero, head, idxn, C], [7 << 16 | OP_PHI_, i32, st, c0, head, stn, C],
                 [4 << 16 | OP_LOOP_MERGE_, M, C, LOOP_CONTROL_DONT_UNROLL], [2 << 16 | OP_BRANCH_, B],
                 [2 << 16 | OP_LABEL_, B]] + loop_body + [[5 << 16 | OP_SHR_LOGICAL_, i32, stn, st, one], [2 << 16 | OP_BRANCH_, C],
                 [2 << 16 | OP_LABEL_, C], [5 << 16 | OP_IEQUAL_, boolean, z, stn, zero], [4 << 16 | OP_BRANCH_COND_, z, M, H],
                 [2 << 16 | OP_LABEL_, M]]
    res, tail = idxn, M
    if cache_var is not None:   # the loop's answer into the cache, and the words of it and the start of the next entry (none past the last: ~0)
        res = nid()
        out.append([3 << 16 | OP_STORE, v_entry, idxn])
        if words:
            ones = m.add_const(word_t, 0xFFFFFFFF)
            lt = em(OP_ULESS_THAN, boolean, idxn, m.add_const(i32, 4095), to=out)
            nxt = em(OP_SELECT_, i32, lt, em(OP_IADD_, i32, idxn, one, to=out), idxn, to=out)
            for k in range(words): out.append([3 << 16 | OP_STORE, v_w[k], table_word(idxn, k, out)])
            for k in (0, 1): out.append([3 << 16 | OP_STORE, v_hi[k], em(OP_SELECT_, word_t, lt, table_word(nxt, k, out), ones, to=out)])
        out += [[2 << 16 | OP_BRANCH_, J], [2 << 16 | OP_LABEL_, J], [7 << 16 | OP_PHI_, i32, res, gc, lab_id, idxn, M]]
        tail = J
    if cache_var is not None and words > 2:   # the table words of the answer read after the search (end, delta) taken from the cache: no load
        mul8, word_at = set(), {}
        for x in post:
            op = x[0] & 0xffff
            if op == OP_IMUL and len(x) == 5 and final in x[3:5] and m.consts.get(x[4] if x[3] == final else x[3]) == (i32, [8]): mul8.add(x[2])
            elif op == OP_IADD_ and len(x) == 5 and (x[3] in mul8 or x[4] in mul8):
                k = m.consts.get(x[4] if x[3] in mul8 else x[3])
                if k is not None and k[0] == i32 and len(k[1]) == 1 and k[1][0] < words: word_at[x[2]] = k[1][0]
            elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and len(x) == 6 and x[3] == map_var and x[5] in word_at: word_at[x[2]] = ('chain', word_at[x[5]])
        for i, x in enumerate(post):
            if x[0] & 0xffff == OP_LOAD and x[1] == word_t and isinstance(word_at.get(x[3]), tuple): post[i] = [4 << 16 | OP_LOAD, x[1], x[2], v_w[word_at[x[3]][1]]]
    for x in post:   # the last select's value is the loop's (or the cached entry's); a phi of a later block that named the split block as its parent now names the last block
        for j in range(1, len(x)):
            if x[j] == final: x[j] = res
        if x[0] & 0xffff == OP_PHI_:
            for j in range(4, len(x), 2):
                if x[j] == lab_id: x[j] = tail
        out.append(x)
    return out


def _cid(m, i32, value, x):
    """the constant operand of the step's IAdd that has this value"""
    for c in x[3:5]:
        k = m.consts.get(c)
        if k is not None and k[0] == i32 and k[1] == [value]: return c
    return None


# ---- the translator's temporaries forwarded within a block (local_forward)
def id_positions(x):
    """Operand positions of a function-body instruction that are ids (maybe_ids' rule, by position; a load's or store's memory access operands left out)"""
    op = x[0] & 0xffff
    if op == OP_LOAD: return (1, 2, 3)
    if op == OP_STORE: return (1, 2)
    if op == OP_EXT_INST: return (1, 2, 3) + tuple(range(5, len(x)))
    if op == OP_COMPOSITE_EXTRACT: return (1, 2, 3)
    if op in (OP_COMPOSITE_INSERT, 79): return (1, 2, 3, 4)
    if op == 251: return (1, 2) + tuple(range(4, len(x), 2))
    mask_at = {87: 5, 88: 5, 91: 5, 92: 5, 95: 5, 98: 5, 89: 6, 90: 6, 93: 6, 94: 6, 96: 6, 97: 6, 99: 4}.get(op)
    if mask_at is not None and len(x) > mask_at: return tuple(range(1, mask_at)) + tuple(range(mask_at + 1, len(x)))
    if op in (OP_SELECTION_MERGE, OP_LOOP_MERGE): return (1, 2) if op == OP_LOOP_MERGE else (1,)
    return tuple(range(1, len(x)))


def local_forward(m):
    """The translator writes every intermediate value to a function variable and reads it back (debug mode, android/SHADERS.md "Size"): tens of thousands of variables
    in a large compute program. Turnip turns them back into values only after nir_opt_copy_prop_vars, which clones its table of them at every if (memory and time that
    grow with both): Ocarina of Time's video network spent 117 s and 700 MB in Turnip's compiler on one program for it, 14 s and 330 MB with this. Within each block a load
    of a variable written (or read) earlier in the block takes that value instead; a variable no load reads any more loses its stores and its declaration. Only variables
    that are loaded and stored directly (no access chain, no call argument) are touched; nothing moves across blocks, so no phi is made. → loads removed"""
    removed, gone = 0, set()
    for f in [x[2] for x in m.ins if x[0] & 0xffff == OP_FUNCTION]:
        a, e = m.function_range(f)
        body = m.ins[a:e + 1]
        local = {x[2] for x in body if x[0] & 0xffff == OP_VARIABLE and len(x) >= 4 and x[3] == SC_FUNCTION}
        if not local: continue
        direct = set(local)
        for x in body:   # (a word of another instruction equal to a variable's id drops the variable: a literal that only looks like it can only keep one unchanged)
            op = x[0] & 0xffff
            if op == OP_VARIABLE: continue
            if op == OP_LOAD:
                if x[2] in direct: direct.discard(x[2])
                if x[1] in direct: direct.discard(x[1])
                if len(x) > 4: direct.difference_update(x[4:])
                continue
            if op == OP_STORE:
                if x[2] in direct: direct.discard(x[2])
                if len(x) > 3: direct.difference_update(x[3:])
                continue
            if not direct.isdisjoint(x[1:]): direct.difference_update(x[1:])
        if not direct: continue
        ren, state, out = {}, {}, []
        def r(v):   # a forwarded load's value (that value may itself be a forwarded load's)
            while v in ren: v = ren[v]
            return v
        for x in body:   # first the forwarding, then every use renamed (a phi of a loop header uses a value defined after it)
            op = x[0] & 0xffff
            if op == OP_LABEL: state = {}
            elif op == OP_STORE and x[1] in direct: state[x[1]] = x[2]
            elif op == OP_LOAD and x[3] in direct:
                if x[3] in state: ren[x[2]] = state[x[3]]; removed += 1; continue
                state[x[3]] = x[2]
            out.append(x)
        if ren:
            for i, x in enumerate(out):
                if x[0] & 0xffff == OP_VARIABLE or ren.keys().isdisjoint(x[1:]): continue
                pos = [p for p in id_positions(x) if x[p] in ren]
                if pos:
                    x = list(x)
                    for p in pos: x[p] = r(x[p])
                    out[i] = x
        read = {x[3] for x in out if x[0] & 0xffff == OP_LOAD}
        dead = direct - read; gone |= dead
        out = [x for x in out if not ((x[0] & 0xffff) == OP_STORE and x[1] in dead) and not ((x[0] & 0xffff) == OP_VARIABLE and x[2] in dead)]
        a, e = m.function_range(f)
        m.ins[a:e + 1] = out
    if removed or gone:
        m.ins = [x for x in m.ins if not (x[0] & 0xffff in (OP_NAME, OP_DECORATE) and x[1] in gone)]   # the removed variables' names
        m.index()
    return removed


# ---- storage buffers read from a base the host writes (ssbo_base)
def ssbo_base(m, c0_var, rows, align, word):
    """NVN's storage buffers are raw global memory: a shader may read before its slot's address (Ocarina of Time's UI text vertex shader reads the glyph records below the
    header record its base instance names, at a negative offset from the address in c0). With the ssbo_align answer the translator indexes a slot's descriptor by
    ((addr_lo + offset) - (addr_lo & -align)) >> 2 (GlobalToStorage, both the inline and the multi-target path), and the host binds the descriptor at addr & ~(align - 1):
    such a read gets a negative index, which robust access turns into zeros (every glyph quad collapsed, no title or menu text). Here each such base, an OpBitwiseAnd of a
    word loaded from the c0 table entry of one of rows (component 0, the address's low word) with -align, becomes a load of component word of the same entry: the low word
    of the guest address where the host's descriptor of that slot starts (dmsh_abi_vk.h DMSH_VK_C0_SSBO_BASE, entry flag DMSH_VK_F_SSBO_BASE), which may lie before the
    address, so the index stays non-negative. The translator loads the c0 word right where it masks it (a Cbuf operand: OpAccessChain c0 0 row component, OpLoad, OpBitcast),
    so that is the only shape taken. A mask of a word of another uniform buffer (SBUF_UBO, an address the game keeps in its own buffer) or of a computed value is not a c0
    slot's base and stays; a mask of any other c0 word (another row, another component, a computed index) is not one the host can serve: the module is then left as it is.
    c0_var: the c0 driver segment's variable (DRVCB reg 0); rows: the vec4 rows of the c0 entries of the shader's SBUF / UBO_PTR bindings; align: ssbo_align; word: the
    component (DMSH_VK_C0_SSBO_BASE / 4). → bases rewritten (0: none, the host binds as before), or -1 when a c0 mask was not one of them (the module unchanged)"""
    if c0_var is None or not rows: return 0
    neg = -align & 0xffffffff
    defs = {x[2]: x for x in m.ins if x[0] & 0xffff in (OP_LOAD, OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN, OP_BITCAST, OP_COPY_OBJECT)}

    def is_mask(c):
        k = m.consts.get(c)
        return k is not None and m.int_type(k[0]) is not None and m.int_type(k[0])[0] == 32 and k[1] == [neg]

    def source(v):   # the c0 word v was loaded from: (row, component, its access chain); 'other' another uniform buffer's word; 'dynamic' c0 read otherwise; None not a load
        for _ in range(8):
            x = defs.get(v)
            if x is None: return None
            op = x[0] & 0xffff
            if op in (OP_BITCAST, OP_COPY_OBJECT): v = x[3]; continue
            if op != OP_LOAD: return None
            p = defs.get(x[3])
            if p is None or p[0] & 0xffff not in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN): return 'dynamic' if x[3] == c0_var else None
            if p[3] != c0_var: return 'other' if m.vars.get(p[3], (0, -1))[1] == SC_UNIFORM else None
            idx = [const_int(m, i) for i in p[4:]]
            return (idx[1], idx[2], p) if len(idx) == 3 and None not in idx else 'dynamic'
        return None

    sites = {}
    for x in m.ins:
        if x[0] & 0xffff != OP_BITWISE_AND or len(x) != 5: continue
        v = x[3] if is_mask(x[4]) else x[4] if is_mask(x[3]) else None
        if v is None: continue
        s = source(v)
        if s is None or s == 'other': continue
        if s == 'dynamic' or s[1] != 0 or s[0] not in rows: return -1
        sites[id(x)] = s[2]
    if not sites: return 0
    comp = {}
    for p in sites.values():   # (constants first: they go in before the functions)
        t = m.consts[p[6]][0]
        if t not in comp: comp[t] = m.add_const(t, word)
    out = []
    for x in m.ins:
        p = sites.get(id(x))
        if p is None: out.append(x); continue
        ptr, ld = m.new_id(), m.new_id(); scalar = m.pointee(p[1])[1]
        out += [[7 << 16 | (p[0] & 0xffff), p[1], ptr, c0_var, p[4], p[5], comp[m.consts[p[6]][0]]], [4 << 16 | OP_LOAD, scalar, ld, ptr],
                [4 << 16 | (OP_COPY_OBJECT if scalar == x[1] else OP_BITCAST), x[1], x[2], ld]]
    m.ins = out
    m.index()
    return len(sites)


# ---- texture-use feedback (tex_feedback)
OP_LOGICAL_OR, OP_INOT_EQUAL, OP_SNEGATE, OP_NOT, OP_UCONVERT, OP_SCONVERT = 166, 171, 126, 200, 113, 114
OP_GNU_ELECT, OP_GNU_BROADCAST_FIRST, OP_ATOMIC_OR, OP_IS_HELPER_INVOCATION = 333, 338, 241, 5381
DEC_ARRAY_STRIDE, DEC_VOLATILE, DEC_OFFSET, BUILTIN_HELPER_INVOCATION, SCOPE_DEVICE = 6, 21, 35, 23, 1
CAP_GROUP_NON_UNIFORM_BALLOT, CAP_DEMOTE = 64, 5379
GLSL_UMIN = 38
TEXFB_PURE = VN_OPS | {OP_SELECT, OP_IEQUAL, OP_INOT_EQUAL, OP_LOGICAL_OR, OP_LOGICAL_AND, OP_LOGICAL_NOT, OP_SNEGATE, OP_NOT, OP_UCONVERT, OP_SCONVERT} | set(range(172, 180))
_NO_TYPE_OPS = {0, 8, 56, 62, 63, 64, 99, 218, 219, 220, 221, 224, 225, 228, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 317, 4416, 5294, 5295, 5364, 5365, 5380}   # no result type
TEXFB_MAX_ID = (1 << 20) - 1   # the pool's last texture ID (dmsh_abi_vk.h DMSH_VK_POOL_TEX_COUNT - 1): the bitmap's last bit


class _SiteValues:
    """Value numbers within one block, for tex_feedback's dedup: two ids with one number hold the same bits in every lane. A load of a Function variable numbers as
    (variable, version), the version moved by any other instruction that takes the variable or a pointer into it (a store, a call, an atomic); a load of an Input,
    uniform block or push constant pointer by the pointer's number; bit casts and copies keep their operand's number; integer and logical arithmetic and GLSL.std.450
    calls of integer result by their operation, type and operands' numbers. Anything else is its own id"""
    def __init__(self, m):
        self.m = m; self.fvars = set(); self.ro = {v for v, (p, sc) in m.vars.items() if sc in (SC_INPUT, SC_PUSH_CONSTANT)
                                                   or (sc == SC_UNIFORM and DEC_BUFFER_BLOCK not in m.decos.get(m.strip_arrays(m.pointee(p)[1])[0], {}))}   # (uniform blocks, not storage)
        self.block()
    def block(self): self.vn, self.root, self.ver = {}, {}, {}
    def __call__(self, i): return self.vn.get(i, i)
    def see(self, x):
        m, op = self.m, x[0] & 0xffff
        if op == OP_LABEL: self.block(); return
        if op == OP_FUNCTION: self.fvars = set(); return
        if op == OP_VARIABLE and len(x) > 3 and x[3] == SC_FUNCTION: self.fvars.add(x[2]); return
        if op == OP_FUNCTION_PARAMETER and m.op(x[1]) == OP_TYPE_POINTER: self.fvars.add(x[2]); return
        if op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN):
            self.root[x[2]] = self.root.get(x[3], x[3]); self.vn[x[2]] = ('ac', self(x[3]), *(self(i) for i in x[4:])); return
        if op == OP_LOAD:
            r = self.root.get(x[3], x[3])
            if r in self.fvars: self.vn[x[2]] = ('ld', self(x[3]), self.ver.get(r, 0))
            elif r in self.ro: self.vn[x[2]] = ('ro', self(x[3]))
            return
        for v in x[1:]:   # anything else that takes a pointer into a Function variable may write it
            r = self.root.get(v, v)
            if r in self.fvars: self.ver[r] = self.ver.get(r, 0) + 1
        if op in (OP_BITCAST, OP_COPY_OBJECT): self.vn[x[2]] = self(x[3])
        elif op in TEXFB_PURE: self.vn[x[2]] = (op, x[1], *(self(i) for i in x[3:]))
        elif op == OP_EXT_INST and m.int_type(m.scalar_of(x[1])): self.vn[x[2]] = ('ext', x[1], x[3], x[4], *(self(i) for i in x[5:]))


def tex_feedback(m, pool_set, pool_binding, fb_binding, spec_id):
    """Texture-use feedback (dmsh_abi_vk.h DMSH_VK_POOL_FB_BINDING, DMSH_VK_SPEC_TEXFB, DMSH_VK_F_TEXFB): before every access chain into the texture pool (the variables
    at set pool_set binding pool_binding: the bindless sampling sites of the translator's GenerateSampledImageLoad) the shader sets bit (id & 31) of word id >> 5 of a
    storage buffer at set pool_set binding fb_binding, id the pool index clamped to 2^20 - 1, when the bool specialization constant spec_id is true (default false: the
    driver folds all of it away) and the invocation is not a helper (fragment). Per site, a waterfall over the distinct IDs of the subgroup: each pass the lanes whose ID is
    the first active lane's elect one lane, which loads the word and only when the bit is clear does the atomicOr (so after the first hit it is a load); the lanes done leave
    the loop. A site whose ID has the same value number as an earlier site's in the same block (the same handle sampled again: PCF taps, several maps of one material) gets
    no code. The block is split at each site: if (on && !helper) { loop { ... } } then the rest of the block; a later block's phi that named the split block names the last
    part. The subgroup operations are right at any subgroup size. → (sites given code, sites in all); (0, 0) and the module unchanged when it reads no pool. ValueError when
    a site cannot be given code (in a loop header block, or the pool used other than through a one-index access chain): the caller keeps the module without feedback"""
    pool = {v for v, d in m.decos.items() if d.get(DEC_SET) == [pool_set] and d.get(DEC_BINDING) == [pool_binding] and v in m.vars}
    if not pool: return 0, 0
    funcs = [x[2] for x in m.ins if x[0] & 0xffff == OP_FUNCTION]
    sites = 0
    for f in funcs:
        a, e = m.function_range(f)
        for x in m.ins[a:e + 1]:
            if pool.isdisjoint(x[1:]) or all(x[j] not in pool for j in id_positions(x)): continue   # (a literal operand that equals a pool id: an ext instruction's number)
            if x[0] & 0xffff in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and x[3] in pool and len(x) == 5 and x[4] not in pool: sites += 1; continue
            raise ValueError(f'texture pool used by opcode {x[0] & 0xffff}, not a one-index access chain')
    if not sites: return 0, 0
    ps = m.model == 'ps'
    helper = None
    if ps and CAP_DEMOTE not in m.caps: helper = input_builtin(m, BUILTIN_HELPER_INVOCATION, m.add_type(OP_TYPE_BOOL))
    u32 = m.add_type(OP_TYPE_INT, 32, 0); boolean = m.add_type(OP_TYPE_BOOL)
    glsl = ensure_glsl(m)
    C = {v: m.add_const(u32, v) for v in (0, 1, 5, 31, TEXFB_MAX_ID, SCOPE_SUBGROUP, SCOPE_DEVICE)}
    rta, st = m.new_id(), m.new_id()   # (a type of its own: its ArrayStride and Block decorations are this variable's)
    for x in ([3 << 16 | OP_TYPE_RUNTIME_ARRAY, rta, u32], [3 << 16 | OP_TYPE_STRUCT, st, rta]): m._insert_decl(x); m.types[x[1]] = x
    pst = m.add_type(OP_TYPE_POINTER, SC_STORAGE_BUFFER, st); pu = m.add_type(OP_TYPE_POINTER, SC_STORAGE_BUFFER, u32)
    fb = m.new_id(); m._insert_decl([4 << 16 | OP_VARIABLE, pst, fb, SC_STORAGE_BUFFER])
    spec = m.new_id(); m._insert_decl([3 << 16 | OP_SPEC_CONSTANT_FALSE, boolean, spec])
    at = next(k for k, y in enumerate(m.ins) if y[0] & 0xffff == OP_DECORATE)
    m.ins[at:at] = [[4 << 16 | OP_DECORATE, rta, DEC_ARRAY_STRIDE, 4], [3 << 16 | OP_DECORATE, st, DEC_BLOCK], [5 << 16 | OP_MEMBER_DECORATE, st, 0, DEC_OFFSET, 0],
                    [4 << 16 | OP_DECORATE, fb, DEC_SET, pool_set], [4 << 16 | OP_DECORATE, fb, DEC_BINDING, fb_binding], [4 << 16 | OP_DECORATE, spec, DEC_SPEC_ID, spec_id]]
    at = next(k for k, y in enumerate(m.ins) if y[0] & 0xffff in (OP_NAME, OP_DECORATE))
    for v, s in ((fb, 'dm_texfb'), (spec, 'dm_texfb_on')): nm = lit_string(s); m.ins.insert(at, [(2 + len(nm)) << 16 | OP_NAME, v, *nm])
    if m.header[1] >= 0x10400:   # SPIR-V 1.4 on: every global variable the entry point uses is in its interface (the translator's modules are 1.3)
        k = next(k for k, y in enumerate(m.ins) if y[0] & 0xffff == OP_ENTRY_POINT); m.ins[k] = m.ins[k] + [fb]; m.ins[k][0] = len(m.ins[k]) << 16 | OP_ENTRY_POINT
    if helper is not None and m.header[1] >= 0x10600 and DEC_VOLATILE not in m.decos.get(helper, {}):   # (1.6: a HelperInvocation variable is Volatile)
        m.ins.insert(next(k for k, y in enumerate(m.ins) if y[0] & 0xffff == OP_DECORATE), [3 << 16 | OP_DECORATE, helper, DEC_VOLATILE])
    caps = [c for c in (CAP_GROUP_NON_UNIFORM, CAP_GROUP_NON_UNIFORM_BALLOT) if c not in m.caps]
    if caps:
        at = max(k for k, x in enumerate(m.ins) if x[0] & 0xffff == OP_CAPABILITY) + 1
        m.ins[at:at] = [[2 << 16 | OP_CAPABILITY, c] for c in caps]
    m.index()
    marked = 0
    for f in funcs:
        a, e = m.function_range(f)
        body = m.ins[a:e + 1]
        headers = set(); lab = None
        for x in body:
            if x[0] & 0xffff == OP_LABEL: lab = x[1]
            elif x[0] & 0xffff == OP_LOOP_MERGE: headers.add(lab)
        tys = {x[2]: x[1] for x in body if x[0] & 0xffff not in _NO_TYPE_OPS and len(x) > 2}
        tys.update((c, t) for c, (t, _) in m.consts.items())
        V = _SiteValues(m); out, last, done = [], {}, set()
        lab = None
        for x in body:
            op = x[0] & 0xffff
            if op == OP_LABEL: lab = cur = x[1]; done = set()
            if not (op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and x[3] in pool):
                V.see(x); out.append(x); continue
            key = V(x[4]); V.see(x)
            if key in done: out.append(x); continue
            if lab in headers: raise ValueError('a texture pool access in a loop header block')
            done.add(key); marked += 1
            nid = m.new_id
            H, LB, T, AT, TM, SM, LC, LM, M = (nid() for _ in range(9))
            def em(op_, ty, *ops):
                r = nid(); out.append([(3 + len(ops)) << 16 | op_, ty, r, *ops]); return r
            iu = x[4] if tys.get(x[4]) == u32 else em(OP_BITCAST, u32, x[4])
            cl = em(OP_EXT_INST, u32, glsl, GLSL_UMIN, iu, C[TEXFB_MAX_ID])
            cond = spec
            if ps:
                hv = em(OP_LOAD, boolean, helper) if helper is not None else em(OP_IS_HELPER_INVOCATION, boolean)
                cond = em(OP_LOGICAL_AND, boolean, spec, em(OP_LOGICAL_NOT, boolean, hv))
            out += [[3 << 16 | OP_SELECTION_MERGE, M, 0], [4 << 16 | OP_BRANCH_CONDITIONAL, cond, H, M],
                    [2 << 16 | OP_LABEL, H], [4 << 16 | OP_LOOP_MERGE, LM, LC, 0], [2 << 16 | OP_BRANCH, LB], [2 << 16 | OP_LABEL, LB]]
            first = em(OP_GNU_BROADCAST_FIRST, u32, C[SCOPE_SUBGROUP], cl)
            eq = em(OP_IEQUAL, boolean, cl, first)
            out += [[3 << 16 | OP_SELECTION_MERGE, SM, 0], [4 << 16 | OP_BRANCH_CONDITIONAL, eq, T, SM], [2 << 16 | OP_LABEL, T]]
            el = em(OP_GNU_ELECT, boolean, C[SCOPE_SUBGROUP])
            bit = em(OP_SHL_LOGICAL, u32, C[1], em(OP_BITWISE_AND, u32, cl, C[31]))
            p = em(OP_ACCESS_CHAIN, pu, fb, C[0], em(OP_SHR_LOGICAL, u32, cl, C[5]))
            clear = em(OP_IEQUAL, boolean, em(OP_BITWISE_AND, u32, em(OP_LOAD, u32, p), bit), C[0])
            go = em(OP_LOGICAL_AND, boolean, el, clear)
            out += [[3 << 16 | OP_SELECTION_MERGE, TM, 0], [4 << 16 | OP_BRANCH_CONDITIONAL, go, AT, TM], [2 << 16 | OP_LABEL, AT]]
            em(OP_ATOMIC_OR, u32, p, C[SCOPE_DEVICE], C[0], bit)
            out += [[2 << 16 | OP_BRANCH, TM], [2 << 16 | OP_LABEL, TM], [2 << 16 | OP_BRANCH, SM], [2 << 16 | OP_LABEL, SM], [2 << 16 | OP_BRANCH, LC],
                    [2 << 16 | OP_LABEL, LC], [4 << 16 | OP_BRANCH_CONDITIONAL, eq, LM, H], [2 << 16 | OP_LABEL, LM], [2 << 16 | OP_BRANCH, M], [2 << 16 | OP_LABEL, M], x]
            last[cur] = M
        if last:
            for x in out:
                if x[0] & 0xffff == OP_PHI:
                    for j in range(4, len(x), 2):
                        if x[j] in last: x[j] = last[x[j]]
        m.ins[a:e + 1] = out
    m.index()
    return marked, sites

