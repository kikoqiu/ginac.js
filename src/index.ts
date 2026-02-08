/**
 * ginac.ts
 * TypeScript wrapper for GiNaC WebAssembly using Manual Bindings.
 */


/**
 * Interface for the Low-Level Emscripten Module (Manual C-API).
 */
interface GiNaCWasmModule {
    // Memory Access
    HEAPU8: Uint8Array;
    HEAP32: Int32Array;

    // Manual Memory Management
    _malloc(size: number): number;
    _free(ptr: number): void;

    // String Helpers (Emscripten built-ins)
    stringToUTF8(str: string, outPtr: number, maxBytes: number): void;
    lengthBytesUTF8(str: string): number;

    // Exported C Functions
    _raw_get_all_exported_functions(): number; // Returns ReturnBox*
    _raw_callFunc(namePtr: number, argsBufPtr: number): number; // Returns ReturnBox*
    _raw_getResult(dataPtr: number, len: number, inputType: number, formatType: number): number; // Returns ReturnBox*
}

// Enum mapping to C++ ResultType
enum ResultType {
    STRING = 1,
    JSON = 2,
    NUMBER = 3,
    MATRIX = 4,
    LATEX = 5
}

export class OperatorBase {
    operatorAdd(right: Param): Expr {
        return op_add(this, right);
    }

    operatorSub(right: Param): Expr {
        return op_subtract(this, right);
    }

    operatorMul(right: Param): Expr {
        return op_multiply(this, right);
    }

    operatorDiv(right: Param): Expr {
        return op_divide(this, right);
    }

    operatorMod(right: Param): Expr {
        return op_mod(this, right);
    }

    operatorPow(right: Param): Expr {
        return op_power(this, right);
    }

    operatorNeg(): Expr {
        return op_negate(this);
    }

    operatorLess(right: Param): Expr {
        return op_less(this, right);
    }

    operatorGreater(right: Param): Expr {
        return op_greater(this, right);
    }

    operatorGreaterEqual(right: Param): Expr {
        return op_greaterequal(this, right);
    }

    operatorLessEqual(right: Param): Expr {
        return op_lessequal(this, right);
    }

    operatorEqual(right: Param): Expr {
        return op_equal(this, right);
    }

    operatorNotEqual(right: Param): Expr {
        return op_notequal(this, right);
    }

}
/**
 * Represents a GiNaC expression.
 * Wraps the underlying binary archive data.
 */
export class Expr extends OperatorBase {
    // Holds the raw binary archive data
    public readonly _data: Uint8Array;
    private readonly _ctx: GiNaCContext;

    constructor(ctx: GiNaCContext, rawData: Uint8Array) {
        super();
        this._ctx = ctx;
        this._data = rawData;
    }

    /**
     * Returns the string representation of the expression.
     */
    public toString(): string {
        return this._ctx._fmt(this._data, ResultType.STRING) as string;
    }

    /**
     * Returns the numerical value.
     * Returns null if the expression cannot be evaluated to a number.
     */
    public toNumber(): number | null {
        return this._ctx._fmt(this._data, ResultType.NUMBER) as number | null;
    }

    /**
     * Returns a JSON object representation of the expression structure.
     */
    public toJSON(): object {
        const jsonStr = this._ctx._fmt(this._data, ResultType.JSON) as string;
        return JSON.parse(jsonStr);
    }

    /**
     * Returns a 2D array of numbers (or strings if symbolic) representing a matrix.
     */
    public toMatrix(): (number | string)[][] {
        return this._ctx._fmt(this._data, ResultType.MATRIX) as (number | string)[][];
    }

    /**
     * Returns the LaTeX representation of the expression.
     */
    public toLatex(): string {
        return this._ctx._fmt(this._data, ResultType.LATEX) as string;
    }
}

export class Symbol extends OperatorBase{
    name: string;
    constructor(name: string){
        super();
        this.name = name;
    }
}

export class SymbolicNumber extends OperatorBase{
    value: string;
    constructor(value: string){
        super();
        this.value = value;
    }
}

export type Param = any | Expr;


export class GiNaCContext {
    private _mod: GiNaCWasmModule;
    private _enc = new TextEncoder();
    private _dec = new TextDecoder();

    constructor(module: any) {
        this._mod = module;
    }

    /**
     * @internal
     * Reads a ReturnBox* from C++ memory.
     */
    private _readBox(boxPtr: number): {
        type: number ;
        len: number ;
        ptr: number ;
    } {
        // boxPtr is in bytes, HEAP32 is Int32 array (4 bytes per element)
        const idx = boxPtr >> 2;
        const heap=this._mod.HEAP32;
        return {
            type: heap[idx]!,
            len:  heap[idx + 1]!,
            ptr:  heap[idx + 2]!
        };
    }

    /**
     * @internal
     * Formats raw binary data to target type by calling C++.
     */
    public _fmt(data: Uint8Array, type: ResultType): any {
        const m = this._mod;
        // Allocate temporary buffer in C++ heap
        const inPtr = m._malloc(data.length);
        m.HEAPU8.set(data, inPtr);

        try {
            // Call C++ conversion
            const boxPtr = m._raw_getResult(inPtr, data.length, 1, type);
            const ret = this._readBox(boxPtr);

            // Handle Error
            if (ret.type === 0) {
                const msg = this._dec.decode(m.HEAPU8.subarray(ret.ptr, ret.ptr + ret.len));
                throw new Error("GiNaC Format Error: " + msg);
            }

            // Handle Number (Type 3)
            if (type === ResultType.NUMBER) {
                if (ret.len === 0) return null;
                // Read 8 bytes double from ptr
                // Use slice to avoid alignment issues if ptr is not multiple of 8 (though malloc usually is)
                const bytes = m.HEAPU8.slice(ret.ptr, ret.ptr + 8);
                return new Float64Array(bytes.buffer)[0];
            }

            // Handle Matrix (Type 4)
            if (type === ResultType.MATRIX) {
                return this._parseMatrix(ret.ptr);
            }

            // Handle Text/JSON (Type 1, 2, 5)
            // Zero-copy decoding using subarray
            return this._dec.decode(m.HEAPU8.subarray(ret.ptr, ret.ptr + ret.len));

        } finally {
            m._free(inPtr);
        }
    }

    /**
     * @internal
     * Parse binary matrix format from C++.
     */
    private _parseMatrix(ptr: number): (number | string)[][] {
        const u8 = this._mod.HEAPU8;
        const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
        
        let off = ptr;
        const rows = dv.getInt32(off, true); off += 4;
        const cols = dv.getInt32(off, true); off += 4;

        const result: (number | string)[][] = [];

        for (let r = 0; r < rows; r++) {
            const row: (number | string)[] = [];
            for (let c = 0; c < cols; c++) {
                const cellType = u8[off++];
                if (cellType === 1) { // Double
                    const val = dv.getFloat64(off, true); off += 8;
                    row.push(val);
                } else { // String
                    const len = dv.getInt32(off, true); off += 4;
                    const str = this._dec.decode(u8.subarray(off, off + len));
                    row.push(str);
                    off += len;
                }
            }
            result.push(row);
        }
        return result;
    }

    /**
     * Executes the C++ function.
     */
    public exec(name: string, ...args: Param[]): Expr {
        const m = this._mod;

        // 1. Serialize arguments into a single buffer
        // Calculate total size first
        let totalSize = 4; // NumArgs
        const prepared = args.map(arg => {
            if (arg instanceof Expr) {
                totalSize += 5 + arg._data.length;
                return { type: 1, data: arg._data };
            } else if(typeof arg === "number"){
                const buf = new Uint8Array(new Float64Array([arg]).buffer);
                totalSize += 5 + buf.length;
                return { type: 2, data: buf };
            } else if (arg instanceof SymbolicNumber) {
                const buf = this._enc.encode(arg.value);
                totalSize += 5 + buf.length;
                return { type: 3, data: buf };
            }else if (arg instanceof Symbol) {
                const buf = this._enc.encode(arg.name);
                totalSize += 5 + buf.length;
                return { type: 4, data: buf };
            }else {
                if(typeof arg !== "string"){
                    arg = String(arg);
                }
                const buf = this._enc.encode(arg);
                totalSize += 5 + buf.length; // Type(1) + Len(4) + Data
                return { type: 0, data: buf };
            }
        });

        const bufPtr = m._malloc(totalSize);
        const u8 = m.HEAPU8;
        const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
        
        let off = bufPtr;
        // Write count
        dv.setInt32(off, prepared.length, true); off += 4;

        // Write args
        for (const p of prepared) {
            u8[off++] = p.type;
            dv.setInt32(off, p.data.length, true); off += 4;
            u8.set(p.data, off);
            off += p.data.length;
        }

        // 2. Prepare function name
        const nameLen = m.lengthBytesUTF8(name) + 1;
        const namePtr = m._malloc(nameLen);
        m.stringToUTF8(name, namePtr, nameLen);

        let resExpr: Expr;

        try {
            // 3. Call C++
            const boxPtr = m._raw_callFunc(namePtr, bufPtr);
            const ret = this._readBox(boxPtr);

            if (ret.type === 0) {
                const msg = this._dec.decode(u8.subarray(ret.ptr, ret.ptr + ret.len));
                throw new Error(`GiNaC Exec Error [${name}]: ${msg}`);
            }

            // 4. Copy result (Binary Archive)
            // Must copy because C++ buffer is reused
            const resData = new Uint8Array(ret.len);
            resData.set(u8.subarray(ret.ptr, ret.ptr + ret.len));
            resExpr = new Expr(this, resData);

        } finally {
            m._free(bufPtr);
            m._free(namePtr);
        }

        return resExpr!;
    }

    /**
     * Returns a list of all exported function names.
     */
    public getExportedFunctions(): any[] {
        const boxPtr = this._mod._raw_get_all_exported_functions();
        const ret = this._readBox(boxPtr);
        // Result is a JSON string
        const json = this._dec.decode(this._mod.HEAPU8.subarray(ret.ptr, ret.ptr + ret.len));
        return JSON.parse(json);
    }
}


export let ginac: GiNaCContext;

/**
 * Initializes the GiNaC WebAssembly module.
 * @param wasmModuleFactory - The function returned by the Emscripten script.
 */
export async function initGiNaC(module:any = undefined): Promise<GiNaCContext> {
    if(ginac){
        return ginac;
    }
    if(!module){
        let createGinacModule: any;
        // @ts-ignore
        const isNode = typeof window === 'undefined';
        if (isNode) {
            // @ts-ignore
            createGinacModule = require("./ginac.js");
        }else{
            // @ts-ignore
            createGinacModule = await importScripts("./ginac.js");
        }    
        module = await createGinacModule();
    }
    ginac = new GiNaCContext(module);
    //console.log("Preloaded exported functions:", ginac.getExportedFunctions());
    return ginac;
}


// ========================================================================
// Generated Helper Functions
// ========================================================================
export function sym(name:string): Symbol {
    return new Symbol(name);
}

export function symNum(value:string|number|bigint): SymbolicNumber {
    return new SymbolicNumber(String(value));
}


/** Computes the characteristic polynomial of a matrix. */
export function charpoly(matrix: Param, variable: Param): Expr {
    return ginac.exec("charpoly", matrix, variable);
}

/** Returns the coefficient of var^deg in expr. */
export function coeff(expr: Param, variable: Param, deg: Param): Expr {
    return ginac.exec("coeff", expr, variable, deg);
}

/** Collects terms involving a variable. */
export function collect(expr: Param, sym: Param): Expr {
    return ginac.exec("collect", expr, sym);
}

/** Collects common factors from the terms of sums. */
export function collect_common_factors(expr: Param): Expr {
    return ginac.exec("collect_common_factors", expr);
}

/** Collects coefficients of a distributed polynomial. */
export function collect_distributed(expr: Param, syms: Param): Expr {
    return ginac.exec("collect_distributed", expr, syms);
}

/** Returns the content of a polynomial (gcd of coefficients). */
export function content(expr: Param, variable: Param): Expr {
    return ginac.exec("content", expr, variable);
}

/** Converts Harmonic polylogarithms to Li functions. */
export function convert_H_to_Li(expr: Param, parameter: Param): Expr {
    return ginac.exec("convert_H_to_Li", expr, parameter);
}

/** Decomposes a rational function. */
export function decomp_rational(expr: Param, variable: Param): Expr {
    return ginac.exec("decomp_rational", expr, variable);
}

/** Returns the degree of the expression with respect to a symbol. */
export function degree(expr: Param, sym: Param): Expr {
    return ginac.exec("degree", expr, sym);
}

/** Returns the denominator of a rational function. */
export function denom(expr: Param): Expr {
    return ginac.exec("denom", expr);
}

/** Computes the determinant of a matrix. */
export function determinant(matrix: Param): Expr {
    return ginac.exec("determinant", matrix);
}

/** Creates a diagonal matrix. */
export function diag(...args: Param[]): Expr {
    return ginac.exec("diag", ...args);
}

/** 
 * Computes the derivative.
 * Supports: diff(expr, symbol) or diff(expr, symbol, order).
 */
export function diff(expr: Param, symbol: Param, order?: Param): Expr {
    if (order !== undefined) {
        return ginac.exec("diff", expr, symbol, order);
    }
    return ginac.exec("diff", expr, symbol);
}

/** Polynomial division (exact). */
export function divide(expr1: Param, expr2: Param): Expr {
    return ginac.exec("divide", expr1, expr2);
}

/** Evaluates an expression numerically. */
export function evalf(expr: Param): Expr {
    return ginac.exec("evalf", expr);
}

/** Evaluates sums, products and integer powers of matrices. */
export function evalm(expr: Param): Expr {
    return ginac.exec("evalm", expr);
}

/** Evaluates integrals. */
export function eval_integ(expr: Param): Expr {
    return ginac.exec("eval_integ", expr);
}

/** Expands an expression. */
export function expand(expr: Param): Expr {
    return ginac.exec("expand", expr);
}

/** Factors a polynomial. */
export function factor(expr: Param): Expr {
    return ginac.exec("factor", expr);
}

/** Finds occurrences of a pattern in an expression. */
export function find(expr: Param, pattern: Param): Expr {
    return ginac.exec("find", expr, pattern);
}

/** Numerical root finding. fsolve(eq, var, start, end). */
export function fsolve(eq: Param, variable: Param, start: Param, end: Param): Expr {
    return ginac.exec("fsolve", eq, variable, start, end);
}

/** Greatest Common Divisor. */
export function gcd(a: Param, b: Param): Expr {
    return ginac.exec("gcd", a, b);
}

/** Checks if expression contains a subexpression. */
export function has(expr: Param, pattern: Param): Expr {
    return ginac.exec("has", expr, pattern);
}

/** Integer content of a polynomial. */
export function integer_content(expr: Param): Expr {
    return ginac.exec("integer_content", expr);
}

/** 
 * Indefinite or definite integral.
 * integral(expr, var) or integral(expr, var, lower, upper).
 */
export function integral( variable: Param, lower: Param, upper: Param, expr: Param): Expr {
    return ginac.exec("integral", variable, lower, upper, expr);
}

/** Inverse of a matrix. */
export function inverse(matrix: Param): Expr {
    return ginac.exec("inverse", matrix);
}

/** Dummy print for tab completion. */
export function iprint(): Expr {
    return ginac.exec("iprint");
}

/** Logic check (returns numeric 1 or 0 inside Expr). */
export function is(relation: Param): Expr {
    return ginac.exec("is", relation);
}

/** Least Common Multiple. */
export function lcm(a: Param, b: Param): Expr {
    return ginac.exec("lcm", a, b);
}

/** Leading coefficient. */
export function lcoeff(expr: Param, sym: Param): Expr {
    return ginac.exec("lcoeff", expr, sym);
}

/** Degree of the leading term. */
export function ldegree(expr: Param, sym: Param): Expr {
    return ginac.exec("ldegree", expr, sym);
}

/** Linear equation solver. */
export function lsolve(eqs: Param, vars: Param): Expr {
    return ginac.exec("lsolve", eqs, vars);
}

/** Map function over operands. */
export function map(expr: Param, funcName: Param): Expr {
    return ginac.exec("map", expr, funcName);
}

/** Pattern matching. */
export function match(expr: Param, pattern: Param): Expr {
    return ginac.exec("match", expr, pattern);
}

/** Number of operands. */
export function nops(expr: Param): Expr {
    return ginac.exec("nops", expr);
}

/** Normalizes a rational function. */
export function normal(expr: Param): Expr {
    return ginac.exec("normal", expr);
}

/** Numerator of a rational function. */
export function numer(expr: Param): Expr {
    return ginac.exec("numer", expr);
}

/** Numerator and Denominator (returns list). */
export function numer_denom(expr: Param): Expr {
    return ginac.exec("numer_denom", expr);
}

/** Get the i-th operand. */
export function op(expr: Param, index: Param): Expr {
    return ginac.exec("op", expr, index);
}

/** Power function (base^exp). */
export function pow(base: Param, exp: Param): Expr {
    return ginac.exec("pow", base, exp);
}

/** Pseudo-remainder. */
export function prem(expr1: Param, expr2: Param, sym: Param): Expr {
    return ginac.exec("prem", expr1, expr2, sym);
}

/** Primitive part of a polynomial. */
export function primpart(expr: Param, sym: Param): Expr {
    return ginac.exec("primpart", expr, sym);
}

export function print(): Expr {
    return ginac.exec("print");
}
export function print_csrc(): Expr {
    return ginac.exec("print_csrc");
}

export function print_latex(): Expr {
    return ginac.exec("print_latex");
}

/** Quotient. */
export function quo(expr1: Param, expr2: Param, sym: Param): Expr {
    return ginac.exec("quo", expr1, expr2, sym);
}

/** Rank of a matrix. */
export function rank(matrix: Param): Expr {
    return ginac.exec("rank", matrix);
}

/** Remainder. */
export function rem(expr1: Param, expr2: Param, sym: Param): Expr {
    return ginac.exec("rem", expr1, expr2, sym);
}

/** Resultant of two polynomials. */
export function resultant(expr1: Param, expr2: Param, sym: Param): Expr {
    return ginac.exec("resultant", expr1, expr2, sym);
}

/** Series expansion. */
export function series(expr: Param, relation: Param, order: Param): Expr {
    return ginac.exec("series", expr, relation, order);
}

/** Converts a series to a polynomial. */
export function series_to_poly(expr: Param): Expr {
    return ginac.exec("series_to_poly", expr);
}

/** Sparse pseudo-remainder. */
export function sprem(expr1: Param, expr2: Param, sym: Param): Expr {
    return ginac.exec("sprem", expr1, expr2, sym);
}

/** 
 * Square-free factorization.
 * Supports sqrfree(expr) or sqrfree(expr, vars_list).
 */
export function sqrfree(expr: Param, vars?: Param): Expr {
    if (vars !== undefined) {
        return ginac.exec("sqrfree", expr, vars);
    }
    return ginac.exec("sqrfree", expr);
}

/** Square-free partial fraction decomposition. */
export function sqrfree_parfrac(expr: Param, sym: Param): Expr {
    return ginac.exec("sqrfree_parfrac", expr, sym);
}

/** Square root. */
export function sqrt(expr: Param): Expr {
    return ginac.exec("sqrt", expr);
}

/** 
 * Substitution. 
 * subs(expr, relation_or_list) or subs(expr, pattern, replacement).
 */
export function subs(expr: Param, arg2: Param, arg3?: Param): Expr {
    if (arg3 !== undefined) {
        return ginac.exec("subs", expr, arg2, arg3);
    }
    return ginac.exec("subs", expr, arg2);
}

/** Trailing coefficient. */
export function tcoeff(expr: Param, sym: Param): Expr {
    return ginac.exec("tcoeff", expr, sym);
}

export function time(): Expr { return ginac.exec("time"); }

/** Trace of a matrix. */
export function trace(matrix: Param): Expr {
    return ginac.exec("trace", matrix);
}

/** Matrix transposition. */
export function transpose(matrix: Param): Expr {
    return ginac.exec("transpose", matrix);
}

/** Unit part. */
export function unit(expr: Param, sym: Param): Expr {
    return ginac.exec("unit", expr, sym);
}

// --- Kernel Functions ---

export function basic_log_kernel(): Expr {
    return ginac.exec("basic_log_kernel");
}

export function multiple_polylog_kernel(arg: Param): Expr {
    return ginac.exec("multiple_polylog_kernel", arg);
}

export function ELi_kernel(n: Param, p: Param, x: Param, y: Param): Expr {
    return ginac.exec("ELi_kernel", n, p, x, y);
}

export function Ebar_kernel(n: Param, p: Param, x: Param, y: Param): Expr {
    return ginac.exec("Ebar_kernel", n, p, x, y);
}

export function Kronecker_dtau_kernel(...args: Param[]): Expr {
    return ginac.exec("Kronecker_dtau_kernel", ...args);
}

export function Kronecker_dz_kernel(...args: Param[]): Expr {
    return ginac.exec("Kronecker_dz_kernel", ...args);
}

export function Eisenstein_kernel(...args: Param[]): Expr {
    return ginac.exec("Eisenstein_kernel", ...args);
}

export function Eisenstein_h_kernel(...args: Param[]): Expr {
    return ginac.exec("Eisenstein_h_kernel", ...args);
}

export function modular_form_kernel(...args: Param[]): Expr {
    return ginac.exec("modular_form_kernel", ...args);
}

export function user_defined_kernel(arg1: Param, arg2: Param): Expr {
    return ginac.exec("user_defined_kernel", arg1, arg2);
}

export function q_expansion_modular_form(arg1: Param, arg2: Param, arg3: Param): Expr {
    return ginac.exec("q_expansion_modular_form", arg1, arg2, arg3);
}



/** Multiple polylogarithm G(a, y) or G(a, s, y). */
export function G(a: Param, y: Param, s?: Param): Expr {
    if (s !== undefined) {
        return ginac.exec("G", a, y, s);
    }
    return ginac.exec("G", a, y);
}

/** Harmonic polylogarithm H(x, y). */
export function H(x: Param, y: Param): Expr {
    return ginac.exec("H", x, y);
}

/** Polylogarithm Li(n, x). */
export function Li(n: Param, x: Param): Expr {
    return ginac.exec("Li", n, x);
}

/** Dilogarithm Li2(x). */
export function Li2(x: Param): Expr {
    return ginac.exec("Li2", x);
}

/** Trilogarithm Li3(x). */
export function Li3(x: Param): Expr {
    return ginac.exec("Li3", x);
}

/** Order term function. */
export function Order(x: Param): Expr {
    return ginac.exec("Order", x);
}

/** Nielsen's generalized polylogarithm S(n, p, x). */
export function S(n: Param, p: Param, x: Param): Expr {
    return ginac.exec("S", n, p, x);
}

/** Absolute value. */
export function abs(x: Param): Expr {
    return ginac.exec("abs", x);
}

/** Inverse cosine. */
export function acos(x: Param): Expr {
    return ginac.exec("acos", x);
}

/** Inverse hyperbolic cosine. */
export function acosh(x: Param): Expr {
    return ginac.exec("acosh", x);
}

/** Inverse sine. */
export function asin(x: Param): Expr {
    return ginac.exec("asin", x);
}

/** Inverse hyperbolic sine. */
export function asinh(x: Param): Expr {
    return ginac.exec("asinh", x);
}

/** Inverse tangent. */
export function atan(x: Param): Expr {
    return ginac.exec("atan", x);
}

/** Inverse tangent of y/x (arctangent with two arguments). */
export function atan2(y: Param, x: Param): Expr {
    return ginac.exec("atan2", y, x);
}

/** Inverse hyperbolic tangent. */
export function atanh(x: Param): Expr {
    return ginac.exec("atanh", x);
}

/** Beta function. */
export function beta(x: Param, y: Param): Expr {
    return ginac.exec("beta", x, y);
}

/** Binomial coefficient. */
export function binomial(n: Param, k: Param): Expr {
    return ginac.exec("binomial", n, k);
}

/** Complex conjugate. */
export function conjugate(x: Param): Expr {
    return ginac.exec("conjugate", x);
}

/** Cosine. */
export function cos(x: Param): Expr {
    return ginac.exec("cos", x);
}

/** Hyperbolic cosine. */
export function cosh(x: Param): Expr {
    return ginac.exec("cosh", x);
}

/** Sign of a complex number (csgn). */
export function csgn(x: Param): Expr {
    return ginac.exec("csgn", x);
}

/** Eta function. */
export function eta(x: Param, y: Param): Expr {
    return ginac.exec("eta", x, y);
}

/** Exponential function. */
export function exp(x: Param): Expr {
    return ginac.exec("exp", x);
}

/** Factorial function. */
export function factorial(n: Param): Expr {
    return ginac.exec("factorial", n);
}

/** Imaginary part of a complex number. */
export function imag_part(x: Param): Expr {
    return ginac.exec("imag_part", x);
}

/** Logarithm of the Gamma function. */
export function lgamma(x: Param): Expr {
    return ginac.exec("lgamma", x);
}

/** Natural logarithm. */
export function log(x: Param): Expr {
    return ginac.exec("log", x);
}

/** Psi function (Digamma) or Polygamma function psi(n, x). */
export function psi(arg0: Param, arg1?: Param): Expr {
    if (arg1 !== undefined) {
        return ginac.exec("psi", arg0, arg1);
    }
    return ginac.exec("psi", arg0);
}

/** Real part of a complex number. */
export function real_part(x: Param): Expr {
    return ginac.exec("real_part", x);
}

/** Sine. */
export function sin(x: Param): Expr {
    return ginac.exec("sin", x);
}

/** Hyperbolic sine. */
export function sinh(x: Param): Expr {
    return ginac.exec("sinh", x);
}

/** Heaviside step function. */
export function step(x: Param): Expr {
    return ginac.exec("step", x);
}

/** Tangent. */
export function tan(x: Param): Expr {
    return ginac.exec("tan", x);
}

/** Hyperbolic tangent. */
export function tanh(x: Param): Expr {
    return ginac.exec("tanh", x);
}

/** Gamma function. */
export function tgamma(x: Param): Expr {
    return ginac.exec("tgamma", x);
}

/** Riemann Zeta function zeta(x) or Hurwitz Zeta zeta(s, x). */
export function zeta(arg0: Param, arg1?: Param): Expr {
    if (arg1 !== undefined) {
        return ginac.exec("zeta", arg0, arg1);
    }
    return ginac.exec("zeta", arg0);
}

/** Derivatives of the Riemann Zeta function. */
export function zetaderiv(n: Param, x: Param): Expr {
    return ginac.exec("zetaderiv", n, x);
}



/**
 * Helper to create an Expr from a raw string without calculation.
 */
export function parse(input: string): Expr {
    return ginac.exec("parse", input); 
}

/**
 * Helper to create a list Expr from multiple inputs.
 */
export function list(... input: Param[]): Expr {
    return ginac.exec("list", ...input); 
}

/**
 * Simplify trigonometric expressions
 */
export function trigsimp(input: Param): Expr {
    return ginac.exec("trigsimp", input); 
}

let integWarn=true;
export function integ(expr: Param, x: Param): Expr {
    if(integWarn){
        integWarn=false;
        console.warn("Integ is unstable feature.");
    }
    return ginac.exec("integ", expr,x); 
}



export function matrix(... input: Param[]): Expr {
    let ls=input[0];
    if(Array.isArray(ls)){
        ls=`{${ls.map(r=>`{${r.join(',')}}`).join(',')}}`;
        console.log(ls);
    }
    return ginac.exec("matrix", ls); 
}


export function op_add(a: Param, b: Param): Expr {
    return ginac.exec("op_add", a, b); 
}

export function op_subtract(a: Param, b: Param): Expr {
    return ginac.exec("op_subtract", a, b);
}
export function op_multiply(a: Param, b: Param): Expr {
    return ginac.exec("op_multiply", a, b); 
}
export function op_divide(a: Param, b: Param): Expr {
    return ginac.exec("op_divide", a, b); 
}
export function op_mod(a: Param, b: Param): Expr {
    return ginac.exec("op_mod", a, b); 
}
export function op_equal(a: Param, b: Param): Expr {
    return ginac.exec("op_equal", a, b); 
}
export function op_notequal(a: Param, b: Param): Expr {
    return ginac.exec("op_notequal", a, b); 
}
export function op_less(a: Param, b: Param): Expr {
    return ginac.exec("op_less", a, b); 
}
export function op_lessequal(a: Param, b: Param): Expr {
    return ginac.exec("op_lessequal", a, b); 
}
export function op_greater(a: Param, b: Param): Expr {
    return ginac.exec("op_greater", a, b); 
}
export function op_greaterequal(a: Param, b: Param): Expr {
    return ginac.exec("op_greaterequal", a, b); 
}
export function op_negate(a: Param): Expr {
    return ginac.exec("op_negate", a); 
}
export function op_power(a: Param, b: Param): Expr {
    return ginac.exec("op_power", a, b); 
}
export function op_factorial(a: Param): Expr {
    return ginac.exec("op_factorial", a); 
}
