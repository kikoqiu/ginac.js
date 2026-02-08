#include <ginac/ginac.h>
using namespace GiNaC;

inline static int ex_to_int(numeric n, int ret=0){
	if(n.is_integer()){
		ret = n.to_int();
	}else if(n.is_real()){
		double val=n.to_double();
		ret = (int) val;
		if((double)ret!=val){
			std::cerr << "Error ex_to_int cast from "<< val <<" to "<< ret << std::endl;
		}
	}
	return ret;
}

void find_all_symbols(const ex& e, exset& syms) {
    if (is_a<symbol>(e)) {
        syms.insert(e);
    } else if (e.nops() > 0) {
        for (size_t i = 0; i < e.nops(); ++i) {
            find_all_symbols(e.op(i), syms);
        }
    }
}

/*
 *  Built-in functions
 */

static ex f_collect(const exprseq &e) {return e[0].collect(e[1]);}
static ex f_collect_distributed(const exprseq &e) {return e[0].collect(e[1], true);}
static ex f_collect_common_factors(const exprseq &e) {return collect_common_factors(e[0]);}
static ex f_convert_H_to_Li(const exprseq &e) {return convert_H_to_Li(e[0], e[1]);}
static ex f_degree(const exprseq &e) {return e[0].degree(e[1]);}
static ex f_denom(const exprseq &e) {return e[0].denom();}
static ex f_evalf(const exprseq &e) {return e[0].evalf();}
static ex f_evalm(const exprseq &e) {return e[0].evalm();}
static ex f_eval_integ(const exprseq &e) {return e[0].eval_integ();}
static ex f_expand(const exprseq &e) {return e[0].expand();}
static ex f_factor(const exprseq &e) {return factor(e[0]);}
static ex f_gcd(const exprseq &e) {return gcd(e[0], e[1]);}
static ex f_has(const exprseq &e) {return e[0].has(e[1]) ? ex(1) : ex(0);}
static ex f_lcm(const exprseq &e) {return lcm(e[0], e[1]);}
static ex f_lcoeff(const exprseq &e) {return e[0].lcoeff(e[1]);}
static ex f_ldegree(const exprseq &e) {return e[0].ldegree(e[1]);}
static ex f_lsolve(const exprseq &e) {return lsolve(e[0], e[1]);}
static ex f_nops(const exprseq &e) {return e[0].nops();}
static ex f_normal(const exprseq &e) {return e[0].normal();}
static ex f_numer(const exprseq &e) {return e[0].numer();}
static ex f_numer_denom(const exprseq &e) {return e[0].numer_denom();}
static ex f_pow(const exprseq &e) {return pow(e[0], e[1]);}
static ex f_sqrt(const exprseq &e) {return sqrt(e[0]);}
static ex f_sqrfree1(const exprseq &e) {return sqrfree(e[0]);}
static ex f_subs2(const exprseq &e) {return e[0].subs(e[1]);}
static ex f_tcoeff(const exprseq &e) {return e[0].tcoeff(e[1]);}

#define CHECK_ARG(num, type, fcn) if (!is_a<type>(e[num])) throw(std::invalid_argument("argument " #num " to " #fcn "() must be a " #type))

static ex f_charpoly(const exprseq &e)
{
	CHECK_ARG(0, matrix, charpoly);
	return ex_to<matrix>(e[0]).charpoly(e[1]);
}

static ex f_coeff(const exprseq &e)
{
	CHECK_ARG(2, numeric, coeff);
	return e[0].coeff(e[1], ex_to_int((ex_to<numeric>(e[2]))));
}

static ex f_content(const exprseq &e)
{
	return e[0].content(e[1]);
}

static ex f_decomp_rational(const exprseq &e)
{
	return decomp_rational(e[0], e[1]);
}

static ex f_determinant(const exprseq &e)
{
	CHECK_ARG(0, matrix, determinant);
	return ex_to<matrix>(e[0]).determinant();
}

static ex f_diag(const exprseq &e)
{
	size_t dim = e.nops();
	matrix &m = *new matrix(dim, dim);
	for (size_t i=0; i<dim; i++)
		m.set(i, i, e.op(i));
	return m;
}

static ex f_diff2(const exprseq &e)
{
	CHECK_ARG(1, symbol, diff);
	return e[0].diff(ex_to<symbol>(e[1]));
}

static ex f_diff3(const exprseq &e)
{
	CHECK_ARG(1, symbol, diff);
	CHECK_ARG(2, numeric, diff);
	return e[0].diff(ex_to<symbol>(e[1]),ex_to_int(ex_to<numeric>(e[2])));
}

static ex f_divide(const exprseq &e)
{
	ex q;
	if (divide(e[0], e[1], q))
		return q;
	else
		return fail();
}

static ex f_find(const exprseq &e)
{
	exset found;
	e[0].find(e[1], found);
	lst l;
	for (auto & i : found)
		l.append(i);
	return l;
}

static ex f_fsolve(const exprseq &e)
{
	CHECK_ARG(1, symbol, fsolve);
	CHECK_ARG(2, numeric, fsolve);
	CHECK_ARG(3, numeric, fsolve);
	return fsolve(e[0], ex_to<symbol>(e[1]), ex_to<numeric>(e[2]), ex_to<numeric>(e[3]));
}

static ex f_integer_content(const exprseq &e)
{
	return e[0].expand().integer_content();
}

static ex f_integral(const exprseq &e)
{
	CHECK_ARG(0, symbol, integral);
	return GiNaC::integral(e[0], e[1], e[2], e[3]);
}

static ex f_inverse(const exprseq &e)
{
	CHECK_ARG(0, matrix, inverse);
	return ex_to<matrix>(e[0]).inverse();
}

static ex f_is(const exprseq &e)
{
	CHECK_ARG(0, relational, is);
	return (bool)ex_to<relational>(e[0]) ? ex(1) : ex(0);
}

class apply_map_function : public map_function {
	ex apply;
public:
	apply_map_function(const ex & a) : apply(a) {}
	virtual ~apply_map_function() {}
	ex operator()(const ex & e) override { return apply.subs(wild() == e, true); }
};

static ex f_map(const exprseq &e)
{
	apply_map_function fcn(e[1]);
	return e[0].map(fcn);
}

static ex f_match(const exprseq &e)
{
	exmap repls;
	if (e[0].match(e[1], repls)) {
		lst repl_lst;
		for (auto & i : repls)
			repl_lst.append(relational(i.first, i.second, relational::equal));
		return repl_lst;
	}
	throw std::runtime_error("FAIL");
}


static ex f_op(const exprseq &e)
{
	CHECK_ARG(1, numeric, op);
	int n=ex_to_int(ex_to<numeric>(e[1]));	
	if (n < 0 || n >= (int)e[0].nops())
		throw(std::out_of_range("second argument to op() is out of range"));
	return e[0].op(n);
}

static ex f_prem(const exprseq &e)
{
	return prem(e[0], e[1], e[2]);
}

static ex f_primpart(const exprseq &e)
{
	return e[0].primpart(e[1]);
}

static ex f_quo(const exprseq &e)
{
	return quo(e[0], e[1], e[2]);
}

static ex f_rank(const exprseq &e)
{
	CHECK_ARG(0, matrix, rank);
	return ex_to<matrix>(e[0]).rank();
}

static ex f_rem(const exprseq &e)
{
	return rem(e[0], e[1], e[2]);
}

static ex f_resultant(const exprseq &e)
{
	CHECK_ARG(2, symbol, resultant);
	return resultant(e[0], e[1], ex_to<symbol>(e[2]));
}

static ex f_series(const exprseq &e)
{
	CHECK_ARG(2, numeric, series);
	return e[0].series(e[1], ex_to_int(ex_to<numeric>(e[2])));
}

static ex f_series_to_poly(const exprseq &e)
{
	CHECK_ARG(0, pseries, series_to_poly);
	return series_to_poly(ex_to<pseries>(e[0]));
}

static ex f_sprem(const exprseq &e)
{
	return sprem(e[0], e[1], e[2]);
}

static ex f_sqrfree2(const exprseq &e)
{
	CHECK_ARG(1, lst, sqrfree);
	return sqrfree(e[0], ex_to<lst>(e[1]));
}

static ex f_sqrfree_parfrac(const exprseq &e)
{
	return sqrfree_parfrac(e[0], ex_to<symbol>(e[1]));
}

static ex f_subs3(const exprseq &e)
{
	CHECK_ARG(1, lst, subs);
	CHECK_ARG(2, lst, subs);
	return e[0].subs(ex_to<lst>(e[1]), ex_to<lst>(e[2]));
}

static ex f_trace(const exprseq &e)
{
	CHECK_ARG(0, matrix, trace);
	return ex_to<matrix>(e[0]).trace();
}

static ex f_transpose(const exprseq &e)
{
	CHECK_ARG(0, matrix, transpose);
	return ex_to<matrix>(e[0]).transpose();
}


static ex f_unit(const exprseq &e)
{
	return e[0].unit(e[1]);
}

static ex f_basic_log_kernel(const exprseq &e)
{
	return basic_log_kernel();	
}

static ex f_multiple_polylog_kernel(const exprseq &e)
{
	return multiple_polylog_kernel(e[0]);	
}

static ex f_ELi_kernel(const exprseq &e)
{
	return ELi_kernel(e[0],e[1],e[2],e[3]);	
}

static ex f_Ebar_kernel(const exprseq &e)
{
	return Ebar_kernel(e[0],e[1],e[2],e[3]);	
}

static ex f_Kronecker_dtau_kernel_4(const exprseq &e)
{
	return Kronecker_dtau_kernel(e[0],e[1],e[2],e[3]);	
}

static ex f_Kronecker_dtau_kernel_3(const exprseq &e)
{
	return Kronecker_dtau_kernel(e[0],e[1],e[2]);	
}

static ex f_Kronecker_dtau_kernel_2(const exprseq &e)
{
	return Kronecker_dtau_kernel(e[0],e[1]);	
}

static ex f_Kronecker_dz_kernel_5(const exprseq &e)
{
	return Kronecker_dz_kernel(e[0],e[1],e[2],e[3],e[4]);	
}

static ex f_Kronecker_dz_kernel_4(const exprseq &e)
{
	return Kronecker_dz_kernel(e[0],e[1],e[2],e[3]);	
}

static ex f_Kronecker_dz_kernel_3(const exprseq &e)
{
	return Kronecker_dz_kernel(e[0],e[1],e[2]);	
}

static ex f_Eisenstein_kernel_6(const exprseq &e)
{
	return Eisenstein_kernel(e[0],e[1],e[2],e[3],e[4],e[5]);	
}

static ex f_Eisenstein_kernel_5(const exprseq &e)
{
	return Eisenstein_kernel(e[0],e[1],e[2],e[3],e[4]);	
}

static ex f_Eisenstein_h_kernel_5(const exprseq &e)
{
	return Eisenstein_h_kernel(e[0],e[1],e[2],e[3],e[4]);	
}

static ex f_Eisenstein_h_kernel_4(const exprseq &e)
{
	return Eisenstein_h_kernel(e[0],e[1],e[2],e[3]);	
}

static ex f_modular_form_kernel_3(const exprseq &e)
{
	return modular_form_kernel(e[0],e[1],e[2]);	
}

static ex f_modular_form_kernel_2(const exprseq &e)
{
	return modular_form_kernel(e[0],e[1]);	
}

static ex f_user_defined_kernel(const exprseq &e)
{
	return user_defined_kernel(e[0],e[1]);	
}

static ex f_q_expansion_modular_form(const exprseq &e)
{
	if ( is_a<Eisenstein_kernel>(e[0]) ) {
		return ex_to<Eisenstein_kernel>(e[0]).q_expansion_modular_form(e[1], ex_to_int(ex_to<numeric>(e[2])));
	}	
	if ( is_a<Eisenstein_h_kernel>(e[0]) ) {
		return ex_to<Eisenstein_h_kernel>(e[0]).q_expansion_modular_form(e[1], ex_to_int(ex_to<numeric>(e[2])));
	}	
	if ( is_a<modular_form_kernel>(e[0]) ) {
		return ex_to<modular_form_kernel>(e[0]).q_expansion_modular_form(e[1], ex_to_int(ex_to<numeric>(e[2])));
	}	
	throw(std::invalid_argument("first argument must be a modular form"));
}

static ex f_dummy(const exprseq &e)
{
	throw(std::logic_error("dummy function called (shouldn't happen)"));
}

static ex f_parse(const exprseq &e)
{
	return e[0];
}

static ex f_list(const exprseq &e)
{
	lst l;
	for (size_t i=0; i<e.nops(); i++)
		l.append(e.op(i));
	return l;
}
static ex f_matrix(const exprseq &e){
	return lst_to_matrix(ex_to<lst>(e[0]));
}
namespace GinacTrig{
	ex trigsimp(const ex &expression);
}

static ex f_trigsimp(const exprseq &e){
	return GinacTrig::trigsimp(e[0]);
}

namespace GiNaC{
	ex eval_integ_ex(const ex & f, const symbol & x);
}
static ex f_integ(const exprseq &e){
	return GiNaC::eval_integ_ex(e[0],ex_to<symbol>(e[1])); 
}
 


/* exp T_EQUAL exp	{$$ = $1 == $3;}
| exp T_NOTEQ exp	{$$ = $1 != $3;}
| exp '<' exp		{$$ = $1 < $3;}
| exp T_LESSEQ exp	{$$ = $1 <= $3;}
| exp '>' exp		{$$ = $1 > $3;}
| exp T_GREATEREQ exp	{$$ = $1 >= $3;}
| exp '+' exp		{$$ = $1 + $3;}
| exp '-' exp		{$$ = $1 - $3;}
| exp '*' exp		{$$ = $1 * $3;}
| exp '/' exp		{$$ = $1 / $3;}
| '-' exp %prec NEG	{$$ = -$2;}
| '+' exp %prec NEG	{$$ = $2;}
| exp '^' exp		{$$ = power($1, $3);}
| exp '!'		{$$ = factorial($1);}
*/
static ex f_op_add(const exprseq &e)
{
	return e[0] + e[1];
}
static ex f_op_subtract(const exprseq &e)
{
    return e[0] - e[1];
}
static ex f_op_multiply(const exprseq &e)
{
    return e[0] * e[1];
}
static ex f_op_divide(const exprseq &e)
{
    return e[0] / e[1];
}

static GiNaC::ex f_op_mod(const GiNaC::exprseq &e)
{
    // Ensure we have exactly two arguments: mod(dividend, divisor)
    if (e.nops() != 2) {
        throw std::runtime_error("mod() requires exactly two arguments");
    }

    // Case 1: Numeric Modulo (Integer Remainder)
    if (GiNaC::is_a<GiNaC::numeric>(e[0]) && GiNaC::is_a<GiNaC::numeric>(e[1])) {
        return GiNaC::irem(GiNaC::ex_to<GiNaC::numeric>(e[0]), 
                           GiNaC::ex_to<GiNaC::numeric>(e[1]));
    }

    // Case 2: Polynomial Remainder
    // We attempt to find a symbol to divide by. 
    // If e[0] contains symbols, we treat it as a polynomial remainder.
    GiNaC::exset syms;
	find_all_symbols(e[0],syms);

    if (!syms.empty()) {
        // Use the first symbol found in the expression for the division
        GiNaC::ex x = *(syms.begin());
        return GiNaC::rem(e[0], e[1], x);
    }

    // Fallback: If no logic matches, return the unevaluated function 
    // (or handle floating point fmod if necessary)
    return GiNaC::indexed(e[0], GiNaC::sy_symm(), e[1]); 
}

static ex f_op_equal(const exprseq &e)
{
    return e[0] == e[1];
}
static ex f_op_notequal(const exprseq &e)
{
    return e[0] != e[1];
}
static ex f_op_less(const exprseq &e)
{
    return e[0] < e[1];
}
static ex f_op_lessequal(const exprseq &e)
{
    return e[0] <= e[1];
}
static ex f_op_greater(const exprseq &e)
{
    return e[0] > e[1];
}
static ex f_op_greaterequal(const exprseq &e)
{
    return e[0] >= e[1];
}
static ex f_op_negate(const exprseq &e)
{
    return -e[0];
}
static ex f_op_power(const exprseq &e)
{
	return power(e[0], e[1]);
}
static ex f_op_factorial(const exprseq &e)
{
	return factorial(e[0]);
}




// ============================================================================
// Type Definitions & Wrapper Wrappers
// ============================================================================

// Function pointer type defined by requirements
typedef ex (*fcnp)(const exprseq &e);

// Table structure
struct fcn_init {
    const char *name;
    fcnp p;
    int num_params;
};




static const fcn_init builtin_fcns[] = {
	{"charpoly", f_charpoly, 2},
	{"coeff", f_coeff, 3},
	{"collect", f_collect, 2},
	{"collect_common_factors", f_collect_common_factors, 1},
	{"collect_distributed", f_collect_distributed, 2},
	{"content", f_content, 2},
	{"convert_H_to_Li", f_convert_H_to_Li, 2},
	{"decomp_rational", f_decomp_rational, 2},
	{"degree", f_degree, 2},
	{"denom", f_denom, 1},
	{"determinant", f_determinant, 1},
	{"diag", f_diag, 0},
	{"diff", f_diff2, 2},
	{"diff", f_diff3, 3},
	{"divide", f_divide, 2},
	{"evalf", f_evalf, 1},
	{"evalm", f_evalm, 1},
	{"eval_integ", f_eval_integ, 1},
	{"expand", f_expand, 1},
	{"factor", f_factor, 1},
	{"find", f_find, 2},
	{"fsolve", f_fsolve, 4},
	{"gcd", f_gcd, 2},
	{"has", f_has, 2},
	{"integer_content", f_integer_content, 1},
	{"integral", f_integral, 4},
	{"inverse", f_inverse, 1},
	{"iprint", f_dummy, 0},      // for Tab-completion
	{"is", f_is, 1},
	{"lcm", f_lcm, 2},
	{"lcoeff", f_lcoeff, 2},
	{"ldegree", f_ldegree, 2},
	{"lsolve", f_lsolve, 2},
	{"map", f_map, 2},
	{"match", f_match, 2},
	{"nops", f_nops, 1},
	{"normal", f_normal, 1},
	{"numer", f_numer, 1},
	{"numer_denom", f_numer_denom, 1},
	{"op", f_op, 2},
	{"pow", f_pow, 2},
	{"prem", f_prem, 3},
	{"primpart", f_primpart, 2},
	{"print", f_dummy, 0},       // for Tab-completion
	{"print_csrc", f_dummy, 0},  // for Tab-completion
	{"print_latex", f_dummy, 0}, // for Tab-completion
	{"quo", f_quo, 3},
	{"rank", f_rank, 1},
	{"rem", f_rem, 3},
	{"resultant", f_resultant, 3},
	{"series", f_series, 3},
	{"series_to_poly", f_series_to_poly, 1},
	{"sprem", f_sprem, 3},
	{"sqrfree", f_sqrfree1, 1},
	{"sqrfree", f_sqrfree2, 2},
	{"sqrfree_parfrac", f_sqrfree_parfrac, 2},
	{"sqrt", f_sqrt, 1},
	{"subs", f_subs2, 2},
	{"subs", f_subs3, 3},
	{"tcoeff", f_tcoeff, 2},
	{"time", f_dummy, 0},        // for Tab-completion
	{"trace", f_trace, 1},
	{"transpose", f_transpose, 1},
	{"unit", f_unit, 2},
	{"basic_log_kernel", f_basic_log_kernel, 0},
	{"multiple_polylog_kernel", f_multiple_polylog_kernel, 1},
	{"ELi_kernel", f_ELi_kernel, 4},
	{"Ebar_kernel", f_Ebar_kernel, 4},
	{"Kronecker_dtau_kernel", f_Kronecker_dtau_kernel_4, 4},
	{"Kronecker_dtau_kernel", f_Kronecker_dtau_kernel_3, 3},
	{"Kronecker_dtau_kernel", f_Kronecker_dtau_kernel_2, 2},
	{"Kronecker_dz_kernel", f_Kronecker_dz_kernel_5, 5},
	{"Kronecker_dz_kernel", f_Kronecker_dz_kernel_4, 4},
	{"Kronecker_dz_kernel", f_Kronecker_dz_kernel_3, 3},
	{"Eisenstein_kernel", f_Eisenstein_kernel_6, 6},
	{"Eisenstein_kernel", f_Eisenstein_kernel_5, 5},
	{"Eisenstein_h_kernel", f_Eisenstein_h_kernel_5, 5},
	{"Eisenstein_h_kernel", f_Eisenstein_h_kernel_4, 4},
	{"modular_form_kernel", f_modular_form_kernel_3, 3},
	{"modular_form_kernel", f_modular_form_kernel_2, 2},
	{"user_defined_kernel", f_user_defined_kernel, 2},
	{"q_expansion_modular_form", f_q_expansion_modular_form, 3},
	{"parse", f_parse, 1},
	{"list", f_list, -1 },
	{"matrix", f_matrix, 1 },
	{"trigsimp", f_trigsimp, 1 },
	{"integ",f_integ,2},
	{"op_add", f_op_add, 2},
	{"op_subtract", f_op_subtract, 2},
	{"op_multiply", f_op_multiply, 2},
	{"op_divide", f_op_divide, 2},
	{"op_mod", f_op_mod, 2},
	{"op_equal", f_op_equal, 2},
	{"op_notequal", f_op_notequal, 2},
	{"op_less", f_op_less, 2},
	{"op_lessequal", f_op_lessequal, 2},
	{"op_greater", f_op_greater, 2},
	{"op_greaterequal", f_op_greaterequal, 2},
	{"op_negate", f_op_negate, 1},
	{"op_power", f_op_power, 2},
	{"op_factorial", f_op_factorial, 1},
	{nullptr, f_dummy, 0}        // End marker
};