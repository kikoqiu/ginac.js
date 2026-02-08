/**
 * integ_ex.cpp
 *
 * Extended Symbolic Integration for GiNaC.
 * Implements a SOTA-inspired integration pipeline based on the Risch Algorithm logic:
 * 1. Preprocessing & Simplification.
 * 2. Polynomial Integration.
 * 3. Rational Function Integration (Hermite Reduction + Rothstein-Trager).
 * 4. Heuristic Risch Algorithm (Transcendental Functions).
 *
 */
#include <emscripten.h>
#include <emscripten/console.h>
#include <sstream>

#include <ginac/ginac.h>
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>
#include <stdexcept>
using namespace GiNaC;

namespace GinacTrig
{
    ex trigsimp(const ex &expression);
}

namespace GiNaC
{

    // Forward declaration of the main entry point
    ex eval_integ_ex(const ex &f, const symbol &x);

    /**
     * Extended Euclidean Algorithm for polynomials in x.
     * Computes s and t such that s*A + t*B = g = gcd(A, B).
     */
    ex gcdex(const ex &A, const ex &B, const symbol &x, ex &s, ex &t)
    {
        // Base case: if B is 0, gcd is A
        if (B.is_zero())
        {
            s = ex(1);
            t = ex(0);
            return A;
        }

        // Initialization for EEA sequence
        // r = remainder, s = coeff of A, t = coeff of B
        // r0 = A = 1*A + 0*B
        // r1 = B = 0*A + 1*B
        ex r0 = A;
        ex r1 = B;
        ex s0 = ex(1);
        ex s1 = ex(0);
        ex t0 = ex(0);
        ex t1 = ex(1);

        while (!r1.is_zero())
        {
            ex q = quo(r0, r1, x);
            ex r_new = r0 - q * r1; // Equivalent to rem(r0, r1, x)

            ex s_new = s0 - q * s1;
            ex t_new = t0 - q * t1;

            // Shift
            r0 = r1;
            r1 = r_new;
            s0 = s1;
            s1 = s_new;
            t0 = t1;
            t1 = t_new;
        }

        // r0 is now the raw GCD computed by EEA
        ex g_raw = r0;
        ex s_raw = s0;
        ex t_raw = t0;

        // Normalization:
        // The GCD returned by standard gcd(A, B) might differ from g_raw by a factor c (independent of x).
        // We calculate c = gcd(A, B) / g_raw to adjust s and t accordingly.
        ex g_canonical = gcd(A, B);
        ex c = quo(g_canonical, g_raw, x);

        s = s_raw * c;
        t = t_raw * c;

        return g_canonical;
    }

    namespace integ
    {

        // ==========================================
        // Section 1: Basic Helpers
        // ==========================================

        bool is_independent(const ex &f, const symbol &x)
        {
            return !f.has(x);
        }

        ex mk_log(const ex &arg)
        {
            return log(arg);
        }

        /**
         * Integrates a single monomial c * x^n.
         */
        ex integrate_monomial(const ex &c, const ex &x_val, int n)
        {
            if (n == -1)
            {
                return c * mk_log(x_val);
            }
            else
            {
                return c * pow(x_val, n + 1) / (n + 1);
            }
        }

        // Helper to check if an expression contains the imaginary unit I
        inline bool has_complex_I(const ex &e)
        {
            return e.has(I);
        }

        // ==========================================
        // Section 2: Polynomial Integration
        // ==========================================

        ex integrate_polynomial(const ex &poly, const symbol &x)
        {
            ex expanded_poly = poly.expand();

            if (is_independent(expanded_poly, x))
            {
                return expanded_poly * x;
            }

            if (is_a<add>(expanded_poly))
            {
                ex res = ex(0);
                for (const auto &term : expanded_poly)
                {
                    res += integrate_polynomial(term, x);
                }
                return res;
            }

            // Handle single term: coeff * x^n
            if (is_a<mul>(expanded_poly))
            {
                ex coeff = ex(1);
                int deg = 0;

                for (const auto &factor : expanded_poly)
                {
                    if (is_independent(factor, x))
                    {
                        coeff *= factor;
                    }
                    else
                    {
                        if (is_a<power>(factor))
                        {
                            if (factor.op(0).is_equal(x))
                            {
                                ex expo = factor.op(1);
                                if (is_a<numeric>(expo) && ex_to<numeric>(expo).is_integer())
                                {
                                    deg += ex_to<numeric>(expo).to_int();
                                }
                                else
                                {
                                    return coeff * pow(x, expo + 1) / (expo + 1);
                                }
                            }
                        }
                        else if (factor.is_equal(x))
                        {
                            deg += 1;
                        }
                    }
                }
                return integrate_monomial(coeff, x, deg);
            }

            if (is_a<power>(expanded_poly))
            {
                if (expanded_poly.op(0).is_equal(x))
                {
                    ex expo = expanded_poly.op(1);
                    if (is_a<numeric>(expo) && ex_to<numeric>(expo).is_integer())
                    {
                        return integrate_monomial(ex(1), x, ex_to<numeric>(expo).to_int());
                    }
                }
            }

            if (expanded_poly.is_equal(x))
            {
                return integrate_monomial(ex(1), x, 1);
            }

            return expanded_poly * x;
        }

        // ==========================================
        // Section 3: Rational Function Integration
        // Logic: Hermite Reduction -> Rothstein-Trager
        // ==========================================

        /**
         * Solves Diophantine equation: A * S + B * T = C (mod 0)
         * We aim for a solution where deg(S) < deg(B).
         */
        bool solve_diophantine(const ex &A, const ex &B, const ex &C, const symbol &x, ex &S, ex &T)
        {
            ex s, t, g;
            // Extended GCD: s*A + t*B = g
            g = GiNaC::gcdex(A, B, x, s, t);

            if (!rem(C, g, x).is_zero())
            {
                return false;
            }

            ex scale = quo(C, g, x);
            ex s_scaled = s * scale;
            // We need deg(S) < deg(B/g). Since B is just V here (and g=1 ideally),
            // we use B directly for modulo if g is constant.

            // General solution structure:
            // S = (s_scaled) mod (B/g)
            ex modulus = quo(B, g, x);
            S = rem(s_scaled, modulus, x);

            // T = (C - A*S) / B
            ex num = C - A * S;
            T = quo(num, B, x);

            return true;
        }

        /**
         * Hermite Reduction.
         * Decomposes Integral(A/D) -> U/V + Integral(NewA/NewD)
         * such that NewD is square-free.
         */
        std::pair<ex, ex> hermite_reduce(const ex &num, const ex &den, const symbol &x)
        {
            ex A = num;
            ex D = den;
            ex int_part = ex(0);

            // Loop to reduce multiplicity of poles in D
            while (true)
            {
                ex D_prime = D.diff(x);
                ex g = gcd(D, D_prime);

                // If g is constant, D is square-free.
                if (is_independent(g, x))
                {
                    return {int_part, A / D};
                }

                // V = D / g. (The square-free part relative to the reduction step)
                ex V = quo(D, g, x);

                // We solve: B * (-V') + C * V = A
                // We want to reduce the order of the pole.
                // Formula: Int(A / (V*g)) = B/g + Int((C - B') / g)
                // provided that -V' and V are coprime (which they are if V is sqrfree).

                ex minus_V_prime = -V.diff(x);
                ex B, C;

                bool solvable = solve_diophantine(minus_V_prime, V, A, x, B, C);

                if (!solvable)
                {
                    // Fallback if algebraic constraints fail (should not happen for rationals)
                    return {int_part, A / D};
                }

                int_part += B / g;

                // Update A and D for next iteration
                A = C - B.diff(x);
                D = g;
            }
        }

        /**
         * Simple polynomial solver for linear and quadratic equations.
         * Returns a list of equations: { var == root1, var == root2, ... }
         * Used by Rothstein-Trager.
         */
        lst solve(const ex &eq, const symbol &var)
        {
            ex poly = (eq.lhs() - eq.rhs()).expand().collect(var);

            if (!poly.is_polynomial(var))
                return lst{};

            int deg = poly.degree(var);

            // Degree 0 (constant != 0) -> no solution
            if (deg < 1)
                return lst{};

            if (deg == 1)
            {
                // ax + b = 0 -> x = -b/a
                ex b = poly.coeff(var, 0);
                ex a = poly.coeff(var, 1);
                return lst{var == -b / a};
            }

            if (deg == 2)
            {
                // ax^2 + bx + c = 0
                ex c = poly.coeff(var, 0);
                ex b = poly.coeff(var, 1);
                ex a = poly.coeff(var, 2);
                ex delta = b * b - 4 * a * c;
                ex r1 = (-b + sqrt(delta)) / (2 * a);
                ex r2 = (-b - sqrt(delta)) / (2 * a);
                // Check for duplicate roots if delta is 0
                if (delta.is_zero())
                {
                    return lst{var == r1};
                }
                return lst{var == r1, var == r2};
            }

            // For higher degrees, we do not implement explicit radical solutions here.
            // Returning empty list implies we cannot integrate this part fully symbolically
            // with the current simple solver.
            return lst{};
        }

        /**
         * Rothstein-Trager Algorithm.
         * Integrates A/D where D is square-free.
         * Returns Sum( r_i * log(D, A - r_i * D') ).
         */
        ex rothstein_trager(const ex &num, const ex &den, const symbol &x)
        {
            symbol t("t");
            ex D_prime = den.diff(x);

            // Resultant wrt x: R(t) = res(D, A - t*D')
            ex poly_for_res = num - t * D_prime;
            ex R = resultant(den, poly_for_res, x);

            // Factorize R(t) over Q
            ex R_factored = factor(R);
            ex result_log = ex(0);

            // Lambda to process each factor of the resultant
            auto process_factor = [&](const ex &fact)
            {
                if (!fact.has(t))
                    return;

                int deg = fact.degree(t);
                if (deg == 1)
                {
                    // Linear factor: t - c
                    ex c = -fact.coeff(t, 0) / fact.coeff(t, 1);
                    ex arg = gcd(den, num - c * D_prime);
                    // Make argument monic to beautify output
                    if (arg.has(x))
                        arg = arg / arg.lcoeff(x);
                    result_log += c * mk_log(arg);
                }
                else
                {
                    // Non-linear factor: algebraic roots.
                    try
                    {
                        // Use our local solve function
                        lst roots = solve(fact == 0, t);

                        // If solver failed (e.g. high degree), warn and skip
                        if (roots.nops() == 0)
                        {
                            std::cout << "[integ_ex] Warning: Algebraic roots not fully resolved in Rothstein-Trager for degree " << deg << "." << std::endl;
                        }

                        for (const auto &eq : roots)
                        {
                            ex r = eq.rhs();
                            ex arg = gcd(den, num - r * D_prime);
                            if (arg.has(x))
                                arg = arg / arg.lcoeff(x);
                            result_log += r * mk_log(arg);
                        }
                    }
                    catch (...)
                    {
                        std::cout << "[integ_ex] Warning: Exception during root solving." << std::endl;
                    }
                }
            };

            if (is_a<mul>(R_factored))
            {
                for (size_t i = 0; i < R_factored.nops(); ++i)
                    process_factor(R_factored.op(i));
            }
            else
            {
                process_factor(R_factored);
            }

            return result_log;
        }

        /**
         * @brief Integrates (Ax+B)/(ax^2+bx+c) directly using real arithmetic.
         * Avoids complex resultants in Rothstein-Trager for simple cases.
         */
        ex integrate_quadratic_log_part(const ex &num, const ex &den, const symbol &x)
        {
            ex a = den.coeff(x, 2);
            ex b = den.coeff(x, 1);
            ex c = den.coeff(x, 0);

            ex A = num.coeff(x, 1);
            ex B = num.coeff(x, 0);

            // 1. Logarithmic part: (A/2a) * log(D)
            // Matches the derivative of the denominator: d(ax^2+bx+c) = 2ax+b
            ex term1 = ex(0);
            if (!A.is_zero())
            {
                term1 = (A / (2 * a)) * log(den);
            }

            // 2. Remainder constant part for arctan/atanh
            // We need to integrate K / (ax^2+bx+c)
            // K comes from the remainder after extracting the log part: B - (A*b)/(2a)
            ex K = B - (A * b) / (2 * a);

            if (K.is_zero())
            {
                return term1;
            }

            ex discriminant = b * b - 4 * a * c;

            // Check discriminant sign to decide between arctan and atanh
            // delta < 0 -> complex roots -> arctan (e.g. 1/(1+x^2))
            // delta > 0 -> real roots    -> atanh  (e.g. 1/(1-x^2))

            if (discriminant.info(info_flags::negative))
            {
                ex h = sqrt(-discriminant); // h = sqrt(4ac - b^2)
                ex arg = (2 * a * x + b) / h;
                return term1 + (2 * K / h) * atan(arg);
            }
            else if (discriminant.info(info_flags::positive))
            {
                ex h = sqrt(discriminant); // h = sqrt(b^2 - 4ac)
                ex arg = (2 * a * x + b) / h;
                return term1 - (2 * K / h) * atanh(arg);
            }
            else
            {
                // Fallback for symbolic discriminant or zero (unlikely in square-free D)
                ex h = sqrt(-discriminant);
                return term1 + (2 * K / h) * atan((2 * a * x + b) / h);
            }
        }

        ex integrate_rational(const ex &f, const symbol &x)
        {
            ex num = f.numer();
            ex den = f.denom();

            // 1. Euclidean Division: f = Poly + Rem/Den
            ex poly_part = quo(num, den, x);
            ex rem_part = rem(num, den, x);

            ex res = ex(0);
            if (!poly_part.is_zero())
            {
                res += integrate_polynomial(poly_part, x);
            }

            if (rem_part.is_zero())
                return res;

            // 2. Make denominator monic
            ex lc = den.lcoeff(x);
            if (!lc.is_equal(ex(1)))
            {
                rem_part = rem_part / lc;
                den = den / lc;
            }

            // 3. Hermite Reduction
            std::pair<ex, ex> hermite_res = hermite_reduce(rem_part, den, x);
            res += hermite_res.first;

            ex A_log = hermite_res.second.numer();
            ex D_log = hermite_res.second.denom(); // This D_log is square-free now

            if (A_log.is_zero())
                return res;

            // 4. Logarithmic Part (Rothstein-Trager or Quadratic Opt)
            if (D_log.degree(x) == 2)
            {
                res += integrate_quadratic_log_part(A_log, D_log, x);
            }
            else
            {
                res += rothstein_trager(A_log, D_log, x);
            }

            return res;
        }

        // ==========================================
        // Section 4: Heuristic Risch Algorithm
        // Logic: Rewrite -> Ansatz -> Diff -> Match -> Solve
        // ==========================================

        ex rewrite_trig_to_exp(const ex &e)
        {
            if (is_a<function>(e))
            {
                std::string name = ex_to<function>(e).get_name();
                ex arg = e.op(0);

                if (name == "sin")
                {
                    return (exp(I * arg) - exp(-I * arg)) / (2 * I);
                }
                else if (name == "cos")
                {
                    return (exp(I * arg) + exp(-I * arg)) / 2;
                }
                else if (name == "tan")
                {
                    ex s = (exp(I * arg) - exp(-I * arg)) / (2 * I);
                    ex c = (exp(I * arg) + exp(-I * arg)) / 2;
                    return s / c;
                }
                else if (name == "sinh")
                {
                    return (exp(arg) - exp(-arg)) / 2;
                }
                else if (name == "cosh")
                {
                    return (exp(arg) + exp(-arg)) / 2;
                }
                else if (name == "tanh")
                {
                    return (exp(arg) - exp(-arg)) / (exp(arg) + exp(-arg));
                }
            }
            return e.map([](const ex &term)
                         { return rewrite_trig_to_exp(term); });
        }

        // Convert exp(I*x) back to cos(x) + I*sin(x)
        ex rewrite_complex_to_trig(const ex &e)
        {
            if (is_a<power>(e))
            {
                ex base = e.op(0);
                ex arg = e.op(1);
                // Detect exp(something)
                if (is_a<numeric>(base) && base.is_equal(GiNaC::exp(1)))
                {   // base is e?
                    // GiNaC represents exp(x) as exp(x), not power(E, x) usually,
                    // but let's handle generic power if E is used.
                    // Check if arg has I
                    if (has_complex_I(arg))
                    {
                        ex real_part = arg / I;
                        // If division by I made it simple (no I left), valid for conversion
                        if (!has_complex_I(real_part))
                        {
                            return cos(real_part) + I * sin(real_part);
                        }
                    }
                }
            }

            // Handle exp() function call
            if (is_a<function>(e) && ex_to<function>(e).get_name() == "exp")
            {
                ex arg = e.op(0);
                // Check if arg has I
                if (has_complex_I(arg))
                {
                    ex real_part = arg / I;
                    // If division by I made it simple (no I left), valid for conversion
                    if (!has_complex_I(real_part))
                    {
                        return cos(real_part) + I * sin(real_part);
                    }
                }
            }

            return e.map([](const ex &term)
                         { return rewrite_complex_to_trig(term); });
        }

        void get_kernels(const ex &f, const symbol &x, std::set<ex, ex_is_less> &kernels)
        {
            if (is_independent(f, x))
                return;

            if (is_a<symbol>(f))
            {
                if (f.is_equal(x))
                    kernels.insert(f);
                return;
            }

            if (is_a<function>(f) || is_a<power>(f))
            {
                // Check for x^n (not a kernel, just x)
                if (is_a<power>(f) && f.op(0).is_equal(x) && is_a<numeric>(f.op(1)))
                {
                    kernels.insert(x);
                    return;
                }
                kernels.insert(f);
                for (size_t i = 0; i < f.nops(); ++i)
                    get_kernels(f.op(i), x, kernels);
                return;
            }

            for (size_t i = 0; i < f.nops(); ++i)
                get_kernels(f.op(i), x, kernels);
        }

        // Helper to extract denominators for logarithmic terms in ansatz
        void get_log_kernels(const ex &f, const symbol &x, std::set<ex, ex_is_less> &log_kernels)
        {
            if (is_independent(f, x))
                return;

            if (is_a<power>(f))
            {
                ex base = f.op(0);
                ex expo = f.op(1);
                if (is_a<numeric>(expo) && ex_to<numeric>(expo).is_negative())
                {
                    // Denominator found: base
                    if (!is_independent(base, x))
                    {
                        log_kernels.insert(base);
                    }
                }
            }

            for (size_t i = 0; i < f.nops(); ++i)
            {
                get_log_kernels(f.op(i), x, log_kernels);
            }
        }

        /**
         * @brief Identifies bases raised to negative integer powers <= -2.
         * e.g. 1/(u^2+1)^2 implies we need terms like 1/(u^2+1) in the ansatz.
         */
        void get_negative_powers(const ex &f, const symbol &x, std::map<ex, int, ex_is_less> &neg_powers)
        {
            if (is_independent(f, x))
                return;

            if (is_a<power>(f))
            {
                ex base = f.op(0);
                ex expo = f.op(1);
                if (is_a<numeric>(expo) && ex_to<numeric>(expo).is_integer())
                {
                    int n = ex_to<numeric>(expo).to_int();
                    if (n <= -2)
                    {
                        if (neg_powers.find(base) == neg_powers.end() || neg_powers[base] > n)
                        {
                            neg_powers[base] = n;
                        }
                    }
                }
            }
            for (size_t i = 0; i < f.nops(); ++i)
                get_negative_powers(f.op(i), x, neg_powers);
        }

        /**
         * @brief Builds the Risch Ansatz.
         */
        ex build_ansatz(const std::set<ex, ex_is_less> &kernels,
                        const std::set<ex, ex_is_less> &log_kernels,
                        const std::map<ex, int, ex_is_less> &neg_powers,
                        const symbol &x,
                        int poly_deg,
                        std::vector<symbol> &coeffs,
                        std::vector<ex> &basis)
        {
            ex ansatz = ex(0);
            basis.clear();

            auto add_term = [&](const ex &term)
            {
                symbol c("C_" + std::to_string(coeffs.size()));
                coeffs.push_back(c);
                basis.push_back(term);
                ansatz += c * term;
            };

            // 1. Polynomial Part: Sum C_i * x^i
            for (int i = 0; i <= poly_deg; ++i)
            {
                add_term(pow(x, i));
            }

            // 2. Kernel parts
            for (const auto &k : kernels)
            {
                if (k.is_equal(x))
                    continue;
                // Generalized heuristic: allow kernel * poly(x)
                for (int i = 0; i <= poly_deg; ++i)
                {
                    add_term(pow(x, i) * k);
                }
            }

            // 3. Log parts
            for (const auto &lk : log_kernels)
            {
                for (int i = 0; i <= poly_deg; ++i)
                {
                    add_term(pow(x, i) * log(lk));
                }
            }

            // 4. Negative Powers
            for (const auto &pair : neg_powers)
            {
                ex base = pair.first;
                int min_pow = pair.second; // e.g. -2
                // We need powers from min_pow + 1 up to -1
                for (int p = min_pow + 1; p <= -1; ++p)
                {
                    for (int i = 0; i <= poly_deg; ++i)
                    {
                        add_term(pow(x, i) * pow(base, p));
                    }
                }
            }

            return ansatz;
        }

        // Corrected helper to extract functional parts of an expression for basis matching
        void get_functional_basis(const ex &expr, const symbol &x, std::set<ex, ex_is_less> &basis)
        {
            if (is_independent(expr, x))
            {
                basis.insert(ex(1));
                return;
            }
            if (GiNaC::is_a<add>(expr))
            {
                for (const auto &term : expr)
                {
                    get_functional_basis(term, x, basis);
                }
            }
            else
            {
                ex functional_part = 1;
                if (GiNaC::is_a<mul>(expr))
                {
                    for (const auto &factor : expr)
                    {
                        if (factor.has(x))
                        {
                            functional_part *= factor;
                        }
                    }
                }
                else
                {
                    functional_part = expr;
                }

                if (functional_part.is_equal(1))
                {
                    basis.insert(ex(1));
                }
                else
                {
                    basis.insert(functional_part);
                }
            }
        }

        ex integrate_heurisch(const ex &f, const symbol &x)
        {
            try
            {
                // 1. Rewrite
                // Use expand() instead of normal() to avoid creating rational functions that confuse kernel extraction
                ex w1 = wild(1), w2 = wild(2);
                ex f_exp = rewrite_trig_to_exp(f).expand().subs(pow(exp(w1), w2) == exp(w1 * w2), subs_options::algebraic).subs(exp(w1) * exp(w2) == exp(w1 + w2), subs_options::algebraic);

                // 1b. Separate constant part (linear term in result)
                // This is crucial for terms like cos(x)^2 -> 1/2 + ...
                // Robustly extract constant terms even if wrapped in mul
                ex f_const = ex(0);
                ex f_var = ex(0);

                if (is_a<add>(f_exp))
                {
                    for (const auto &term : f_exp)
                    {
                        if (is_independent(term, x))
                            f_const += term;
                        else
                            f_var += term;
                    }
                }
                else
                {
                    if (is_independent(f_exp, x))
                        f_const = f_exp;
                    else
                        f_var = f_exp;
                }

                ex res_const = f_const * x;

                if (f_var.is_zero())
                {
                    return res_const;
                }

                // 2. Identify Heuristic Degree
                int x_deg = 0;
                if (f_var.has(x))
                {
                    if (f_var.is_polynomial(x))
                    {
                        x_deg = f_var.degree(x);
                    }
                    else
                    {
                        // Dynamic degree approximation for mixed terms
                        x_deg = 2;
                    }
                }

                // 3. Extract Kernels & Features
                std::set<ex, ex_is_less> kernels;
                get_kernels(f_var, x, kernels);
                kernels.erase(x); // x is the integration variable, not a kernel here.

                // 2b. Log candidates (denominators)
                std::set<ex, ex_is_less> log_kernels;
                get_log_kernels(f_var, x, log_kernels);

                std::map<ex, int, ex_is_less> neg_powers;
                get_negative_powers(f_var, x, neg_powers);

                // 4. Build Ansatz
                std::vector<symbol> coeffs;
                std::vector<ex> ansatz_basis;
                ex ansatz = build_ansatz(kernels, log_kernels, neg_powers, x, x_deg, coeffs, ansatz_basis);

                // 5. Diff & Error
                ex diff_ansatz = ansatz.diff(x).normal();
                ex error = (diff_ansatz - f_var).normal();
                ex num_error = error.numer();

                // 6. Symbolic Linear System Generation
                std::set<ex, ex_is_less> match_basis_set;
                get_functional_basis(num_error, x, match_basis_set);

                std::vector<ex> match_basis(match_basis_set.begin(), match_basis_set.end());
                // Sort basis to be deterministic, putting more complex terms first.
                std::sort(match_basis.begin(), match_basis.end(), [](const ex &a, const ex &b)
                          {
                              return ex_is_less()(b, a); // Sort descending
                          });

                lst eq_list;
                ex processing_error = num_error;

                for (const auto &basis_func : match_basis)
                {
                    if (processing_error.is_zero())
                        break;
                    // Important: Do not skip basis_func == 1 here.
                    // We need to match constant errors to polynomial coefficients in ansatz (e.g. c1*x -> c1).

                    ex collected = processing_error.collect(basis_func);
                    ex coeff = collected.coeff(basis_func, 1);

                    if (!coeff.is_zero())
                    {
                        // The coefficient must be zero for ALL x.
                        ex poly_coeff = coeff.expand();
                        if (is_independent(poly_coeff, x))
                        {
                            eq_list.append(poly_coeff == 0);
                        }
                        else
                        {
                            int deg = poly_coeff.degree(x);
                            int ldeg = poly_coeff.ldegree(x);
                            for (int i = ldeg; i <= deg; ++i)
                            {
                                ex c = poly_coeff.coeff(x, i);
                                if (!c.is_zero())
                                    eq_list.append(c == 0);
                            }
                        }

                        // Subtract the part that was matched. expand() is important.
                        processing_error = (collected.coeff(basis_func, 0)).expand();
                    }
                }

                // The final remaining part must also be zero.
                if (!processing_error.is_zero())
                {
                    ex rem_poly = processing_error.expand();
                    if (is_independent(rem_poly, x))
                    {
                        eq_list.append(rem_poly == 0);
                    }
                    else
                    {
                        int deg = rem_poly.degree(x);
                        int ldeg = rem_poly.ldegree(x);
                        for (int i = ldeg; i <= deg; ++i)
                        {
                            ex c = rem_poly.coeff(x, i);
                            if (!c.is_zero())
                                eq_list.append(c == 0);
                        }
                    }
                }

                lst var_list;
                for (const auto &c : coeffs)
                    var_list.append(c);

                // 7. Solve
                ex sol = lsolve(eq_list, var_list);

                if (sol.nops() == 0 && f.has(x))
                {
                    return integral(x, ex(0), x, f);
                }

                // 8. Subst back
                ex res = ansatz.subs(sol);

                // Clean up any free variables (coefficients that could be anything) by setting them to 0.
                for (const auto &c : coeffs)
                {
                    if (res.has(c))
                        res = res.subs(c == 0);
                }

                return res_const + res;
            }
            catch (...)
            {
                return integral(x, ex(0), x, f);
            }
        }

    } // namespace integ

    /**
     * @brief Helper class to split constants inside logarithms.
     *
     * GiNaC does not automatically expand log(a*b) to log(a) + log(b) due to
     * complex branch cut rules. This map_function forces the split for
     * simplification purposes, e.g., log(2*x) -> log(2) + log(x).
     */
    class LogConstSplitter : public map_function
    {
        symbol x; // The integration variable
    public:
        LogConstSplitter(symbol var) : x(var) {}

        ex operator()(const ex &e) override
        {
            // 1. Check if the expression is a function and specifically 'log'
            if (is_a<function>(e) && is_ex_the_function(e, log))
            {
                ex arg = e.op(0);

                // 2. Check if the argument is a multiplication object (mul)
                if (is_a<mul>(arg))
                {
                    ex const_part = 1;
                    ex var_part = 1;

                    // 3. Iterate through factors to separate constants from variables
                    for (size_t i = 0; i < arg.nops(); ++i)
                    {
                        ex factor = arg.op(i);
                        if (!factor.has(x))
                        {
                            const_part *= factor; // Accumulate constants
                        }
                        else
                        {
                            var_part *= factor; // Accumulate variable parts
                        }
                    }

                    // 4. If there is a non-trivial constant part, split the log
                    if (const_part != 1)
                    {
                        // Result: log(constant) + log(variable_part)
                        // We recursively apply (*this) to the variable part to handle nested cases
                        return log(const_part) + (*this)(log(var_part));
                    }
                }
            }

            // Apply recursively to sub-expressions
            return e.map(*this);
        }
    };

    /**
     * @brief Removes integration constants and cleans up logarithms.
     *
     * @param e The expression to clean.
     * @param x The integration variable.
     * @return ex The cleaned expression without additive constants.
     */
    ex clean_integration_constants(ex e, symbol x)
    {
        // Step 1: Split logs with internal constants
        // e.g., -log(2*sin(x)) becomes -1 * (log(2) + log(sin(x)))
        LogConstSplitter splitter(x);
        e = splitter(e);

        // Step 2: Expand the expression
        // This distributes coefficients: -log(2) - log(sin(x))
        e = e.expand();

        // Step 3: Remove terms that do not contain the variable x
        if (!e.has(x))
        {
            // If the entire expression is constant, return 0
            return 0;
        }

        if (is_a<add>(e))
        {
            exvector new_terms;
            // Iterate over all terms in the sum
            for (size_t i = 0; i < e.nops(); ++i)
            {
                ex term = e.op(i);
                // Keep the term only if it depends on x
                if (term.has(x))
                {
                    new_terms.push_back(term);
                }
            }
            // Reconstruct the expression with filtered terms
            return add(new_terms);
        }

        // If it's not an 'add' object (e.g., a single monomial like sin(x)), return as is
        return e;
    }

    // ==========================================
    // Main Dispatcher
    // ==========================================

    ex eval_integ_ex(const ex &f, const symbol &x)
    {
        bool original_has_I = integ::has_complex_I(f);

        ex expr = f.expand();
        ex res;

        // 1. Linear Sums
        if (is_a<add>(expr))
        {
            ex sum_res = ex(0);
            for (const auto &term : expr)
            {
                sum_res += eval_integ_ex(term, x);
            }
            res = sum_res;
        }
        // 2. Constant Factors
        else if (is_a<mul>(expr))
        {
            ex coeff = ex(1);
            ex integrand = ex(1);
            for (const auto &term : expr)
            {
                if (integ::is_independent(term, x))
                    coeff *= term;
                else
                    integrand *= term;
            }
            if (coeff != ex(1))
                res = coeff * eval_integ_ex(integrand, x);
            else
                goto compute; // Fallthrough
        }
        // 3. Independent
        else if (integ::is_independent(expr, x))
        {
            res = expr * x;
        }
        else
        {
        compute:
            // 4. Rational Functions
            if (expr.info(info_flags::rational_function))
            {
                if (expr.denom().is_equal(ex(1)))
                {
                    res = integ::integrate_polynomial(expr, x);
                }
                else
                {
                    res = integ::integrate_rational(expr, x);
                }
            }
            // 5. Transcendental / Heuristic
            else
            {
                res = integ::integrate_heurisch(expr, x);
                res = GinacTrig::trigsimp(res);
            }
        }

        return clean_integration_constants(res.normal(), x);
    }

} // namespace GiNaC