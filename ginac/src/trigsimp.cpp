/**
 * GinacTrig: Industrial-Strength Trigonometric Simplification
 * Based on Fu's Algorithm (2006): "Automated Simplification of Trigonometric Expressions"
 *
 * This file contains the complete implementation of the simplification strategies (TR1-TR12)
 * and the heuristic driver loop.
 */
#include <emscripten.h>
#include <emscripten/console.h>
#include <sstream>
#include <ginac/ginac.h>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <limits>

using namespace GiNaC;

namespace GinacTrig
{

    // =========================================================================
    // SECTION 1: Infrastructure & Complexity Metrics
    // =========================================================================

    /**
     * Complexity Visitor
     * Evaluates the "cost" of an expression. Lower is simpler.
     * Inherits from specific visitors as requested to ensure correct overload resolution.
     */
    class complexity_visitor : public visitor, public numeric::visitor, public symbol::visitor, public add::visitor, public mul::visitor, public power::visitor, public function::visitor, public basic::visitor
    {
    public:
        long score = 0;

        void visit(const numeric &n)
        {
            if (n.is_real())
            {
                score += 1;
            }
            else
            {
                score += 2;
            }
        }

        void visit(const symbol &s)
        {
            score += 3;
        }

        void visit(const add &a)
        {
            score += 3 * (a.nops() - 1); // Penalize number of terms
            for (size_t i = 0; i < a.nops(); ++i)
                a.op(i).accept(*this);
        }

        void visit(const mul &m)
        {
            score += 4 * (m.nops() - 1); // Penalize number of factor
            for (size_t i = 0; i < m.nops(); ++i)
                m.op(i).accept(*this);
        }

        void visit(const power &p)
        {
            score += 8;
            p.op(0).accept(*this);
            p.op(1).accept(*this);
        }

        void visit(const function &f)
        {
            score += 16;
            for (size_t i = 0; i < f.nops(); ++i)
                f.op(i).accept(*this);
        }

        void visit(const basic &b)
        {
            score += b.nops();
            for (size_t i = 0; i < b.nops(); ++i)
                b.op(i).accept(*this);
        }
    };

    long get_complexity(const ex &e)
    {
        complexity_visitor v;
        e.accept(v);
        return v.score;
    }

    /**
     * Helper: Extract unique arguments from trigonometric functions.
     * Used to target specific variables for expansion/contraction.
     */
    void gather_trig_args(const ex &e, std::set<ex, ex_is_less> &args)
    {
        if (is_a<function>(e))
        {
            std::string name = ex_to<function>(e).get_name();
            if (name == "sin" || name == "cos" || name == "tan" || name == "cot" ||
                name == "sec" || name == "csc" ||
                name == "sinh" || name == "cosh" || name == "tanh")
            {
                args.insert(e.op(0));
            }
        }
        for (size_t i = 0; i < e.nops(); ++i)
            gather_trig_args(e.op(i), args);
    }

    /**
     * Helper: Greedy Update
     * Updates best_expr if candidate has a strictly lower complexity score.
     * Returns true if updated.
     */
    bool greedy_update(ex &best_expr, long &min_score, const ex &candidate)
    {
        // We normalize candidate to combine fractions before measuring
        ex clean = candidate.normal();
        long score = get_complexity(clean);

        if (score < min_score)
        {
            min_score = score;
            best_expr = clean;
            return true;
        }
        return false;
    }

    // =========================================================================
    // SECTION 2: The Strategies (Operators)
    // =========================================================================

    // --- TR1: Functional Normalization ---
    // Converts tan, cot, sec, csc -> sin, cos.
    ex TR1_normalize(const ex &e)
    {
        struct map_tr1 : public map_function
        {
            ex operator()(const ex &e) override
            {
                if (is_a<function>(e))
                {
                    std::string name = ex_to<function>(e).get_name();
                    ex arg = e.op(0);
                    if (name == "tan")
                        return sin(arg) / cos(arg);
                    if (name == "cot")
                        return cos(arg) / sin(arg);
                    if (name == "sec")
                        return 1 / cos(arg);
                    if (name == "csc")
                        return 1 / sin(arg);
                    if (name == "tanh")
                        return sinh(arg) / cosh(arg);
                    if (name == "coth")
                        return cosh(arg) / sinh(arg);
                    if (name == "sech")
                        return 1 / cosh(arg);
                    if (name == "csch")
                        return 1 / sinh(arg);
                }
                return e.map(*this);
            }
        };
        map_tr1 mapper;
        return mapper(e).normal();
    }

    // --- TR2: Expansion of Trig Sums ---
    // sin(x+y) -> sin(x)cos(y) + cos(x)sin(y)
    // cos(x+y) -> cos(x)cos(y) - sin(x)sin(y)
    ex TR2_expand_trig_sums(const ex &e_in)
    {
        struct expand_sums_map : public map_function
        {
            ex operator()(const ex &e) override
            {
                if (is_a<function>(e))
                {
                    std::string name = ex_to<function>(e).get_name();
                    ex arg = e.op(0);

                    // Handle sin(A+B), cos(A+B)
                    if (is_a<add>(arg) && (name == "sin" || name == "cos"))
                    {
                        ex x = arg.op(0);
                        ex y = arg - x; // The rest of the sum

                        if (name == "sin")
                        {
                            return (sin(x) * cos(y) + cos(x) * sin(y)).map(*this);
                        }
                        else if (name == "cos")
                        {
                            return (cos(x) * cos(y) - sin(x) * sin(y)).map(*this);
                        }
                    }

                    // Handle sinh(A+B), cosh(A+B)
                    if (is_a<add>(arg) && (name == "sinh" || name == "cosh"))
                    {
                        ex x = arg.op(0);
                        ex y = arg - x;

                        if (name == "sinh")
                        {
                            return (sinh(x) * cosh(y) + cosh(x) * sinh(y)).map(*this);
                        }
                        else if (name == "cosh")
                        {
                            return (cosh(x) * cosh(y) + sinh(x) * sinh(y)).map(*this);
                        }
                    }
                }
                return e.map(*this);
            }
        };
        expand_sums_map mapper;
        return mapper(e_in);
    }

    // --- TR5: Pythagorean Simplification (Enhanced) ---
    // Replaces sin^2 with 1-cos^2, cosh^2 with 1+sinh^2, etc.
    ex TR5_pythagoras(const ex &e_in)
    {
        ex current = e_in;
        std::set<ex, ex_is_less> args;
        gather_trig_args(current, args);

        if (args.empty())
            return current;

        for (const auto &arg : args)
        {
            // 1. Standard Trig: sin^2 + cos^2 = 1
            ex s2 = pow(sin(arg), 2);
            ex c2 = pow(cos(arg), 2);

            if (current.has(s2) || current.has(c2))
            {
                // Strategy A: Eliminate sin^2
                ex expr_cos = current.subs(s2 == 1 - c2, subs_options::algebraic).expand();
                // Strategy B: Eliminate cos^2
                ex expr_sin = current.subs(c2 == 1 - s2, subs_options::algebraic).expand();

                long score_orig = get_complexity(current);
                long score_cos = get_complexity(expr_cos);
                long score_sin = get_complexity(expr_sin);

                if (score_cos < score_orig && score_cos <= score_sin)
                {
                    current = expr_cos;
                }
                else if (score_sin < score_orig)
                {
                    current = expr_sin;
                }
            }

            // 2. Hyperbolic Trig: cosh^2 - sinh^2 = 1
            ex sh2 = pow(sinh(arg), 2);
            ex ch2 = pow(cosh(arg), 2);

            if (current.has(sh2) || current.has(ch2))
            {
                // Strategy A: Eliminate sinh^2 -> ch2 - 1
                ex expr_ch = current.subs(sh2 == ch2 - 1, subs_options::algebraic).expand();
                // Strategy B: Eliminate cosh^2 -> sh2 + 1
                ex expr_sh = current.subs(ch2 == sh2 + 1, subs_options::algebraic).expand();

                long score_curr = get_complexity(current);
                long score_ch = get_complexity(expr_ch);
                long score_sh = get_complexity(expr_sh);

                if (score_ch < score_curr && score_ch <= score_sh)
                {
                    current = expr_ch;
                }
                else if (score_sh < score_curr)
                {
                    current = expr_sh;
                }
            }
        }
        return current;
    }

    // --- TR7: Power Reduction ---
    // Reduces sin(x)^n, cos(x)^n (n>=2) to multiple angle terms.
    ex TR7_power_reduction(const ex &e_in)
    {
        ex current = e_in;
        bool changed = true;
        int limit = 3;

        struct power_reducer : public map_function
        {
            bool &modified;
            power_reducer(bool &m) : modified(m) {}

            ex operator()(const ex &e) override
            {
                if (is_a<power>(e))
                {
                    ex base = e.op(0);
                    ex exp = e.op(1);

                    if (exp.info(info_flags::posint) && exp > 1)
                    {
                        if (is_a<function>(base))
                        {
                            std::string name = ex_to<function>(base).get_name();
                            ex arg = base.op(0);

                            if (name == "sin")
                            {
                                modified = true;
                                return (pow(base, exp - 2) * (1 - cos(2 * arg)) / 2).expand().map(*this);
                            }
                            if (name == "cos")
                            {
                                modified = true;
                                return (pow(base, exp - 2) * (1 + cos(2 * arg)) / 2).expand().map(*this);
                            }
                        }
                    }
                }
                return e.map(*this);
            }
        };

        while (changed && limit-- > 0)
        {
            changed = false;
            power_reducer reducer(changed);
            current = reducer(current);
        }
        return current.expand();
    }

    // --- TR8: Product-to-Sum ---
    // sin(A)cos(B) -> 1/2(sin(A+B)+sin(A-B))
    struct map_tr8 : public map_function
    {
        ex operator()(const ex &e) override
        {
            if (is_a<mul>(e))
            {
                std::vector<ex> factors;
                for (size_t i = 0; i < e.nops(); ++i)
                    factors.push_back(e.op(i));

                for (auto it1 = factors.begin(); it1 != factors.end(); ++it1)
                {
                    if (!is_a<function>(*it1))
                        continue;
                    std::string n1 = ex_to<function>(*it1).get_name();
                    if (n1 != "sin" && n1 != "cos")
                        continue;

                    for (auto it2 = it1 + 1; it2 != factors.end(); ++it2)
                    {
                        if (!is_a<function>(*it2))
                            continue;
                        std::string n2 = ex_to<function>(*it2).get_name();
                        if (n2 != "sin" && n2 != "cos")
                            continue;

                        ex A = it1->op(0);
                        ex B = it2->op(0);
                        ex replacement;

                        if (n1 == "sin" && n2 == "sin")
                        {
                            replacement = (cos(A - B) - cos(A + B)) / 2;
                        }
                        else if (n1 == "cos" && n2 == "cos")
                        {
                            replacement = (cos(A - B) + cos(A + B)) / 2;
                        }
                        else
                        {
                            if (n1 == "cos")
                            {
                                std::swap(A, B);
                            } // Ensure sin(A)cos(B)
                            replacement = (sin(A + B) + sin(A - B)) / 2;
                        }

                        ex rest = ex(1);
                        for (auto it = factors.begin(); it != factors.end(); ++it)
                        {
                            if (it != it1 && it != it2)
                                rest *= (*it);
                        }
                        return ((*this)(rest * replacement)).expand();
                    }
                }
            }
            return e.map(*this);
        }
    };

    ex TR8_product_to_sum(const ex &e)
    {
        map_tr8 mapper;
        return mapper(e);
    }

    // --- TR9: Sum-to-Product ---
    ex TR9_sum_to_product(const ex &e)
    {
        if (!is_a<add>(e))
            return e;

        ex current = e;
        std::vector<ex> terms;
        for (size_t i = 0; i < current.nops(); ++i)
            terms.push_back(current.op(i));

        for (auto it1 = terms.begin(); it1 != terms.end(); ++it1)
        {
            if (!is_a<function>(*it1))
                continue;
            std::string n1 = ex_to<function>(*it1).get_name();
            if (n1 != "sin" && n1 != "cos")
                continue;

            for (auto it2 = it1 + 1; it2 != terms.end(); ++it2)
            {
                if (!is_a<function>(*it2))
                    continue;
                std::string n2 = ex_to<function>(*it2).get_name();
                if (n1 != n2)
                    continue;

                ex A = it1->op(0);
                ex B = it2->op(0);
                ex res;

                if (n1 == "sin")
                    res = 2 * sin((A + B) / 2) * cos((A - B) / 2);
                else
                    res = 2 * cos((A + B) / 2) * cos((A - B) / 2);

                ex remainder = ex(0);
                for (auto it = terms.begin(); it != terms.end(); ++it)
                {
                    if (it != it1 && it != it2)
                        remainder += *it;
                }
                return (remainder + res);
            }
        }
        return current;
    }

    // Function to normalize trigonometric arguments and simplify expressions.
    // It handles cases like sin(-x) -> -sin(x) and cos(-2*x) -> cos(2*x),
    // allowing terms like 1/2*sin(x) and -1/2*sin(-x) to properly cancel out.
    ex minus_angle(const ex &e_in)
    { // Function to normalize trigonometric arguments and simplify expressions.
        // Map function to enforce parity rules on trigonometric functions.
        struct parity_normalizer : public map_function
        {
            ex operator()(const ex &e) override
            {
                // Check if the expression is a function object
                if (is_a<function>(e))
                {
                    std::string name = ex_to<function>(e).get_name();

                    // Target standard trigonometric functions
                    if (name == "sin" || name == "cos" || name == "tan")
                    {
                        ex arg = e.op(0);
                        ex coeff = 1;

                        // Strategy: Extract the overall numeric coefficient of the argument.
                        // This handles monomials like -x (coeff -1) or -2*x (coeff -2).
                        if (is_a<numeric>(arg))
                        {
                            coeff = arg;
                        }
                        else if (is_a<mul>(arg))
                        {
                            // Iterate through the product terms to find numeric factors.
                            // In GiNaC, -2*x is stored as mul(-2, x).
                            for (size_t i = 0; i < arg.nops(); ++i)
                            {
                                if (is_a<numeric>(arg.op(i)))
                                {
                                    coeff *= arg.op(i);
                                }
                            }
                        }
                        else
                        {
                            if (is_a<add>(arg))
                            {
                                // Get the first term of the addition
                                ex first_term = arg.op(0);
                                if (is_a<mul>(first_term))
                                {
                                    for (size_t i = 0; i < first_term.nops(); ++i)
                                    {
                                        if (is_a<numeric>(first_term.op(i)))
                                        {
                                            if (first_term.op(i).info(info_flags::negative))
                                            {
                                                coeff *= ex(-1);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // If the numeric coefficient is negative, apply parity identities.
                        // using .info(info_flags::real) ensures we are dealing with real numbers.
                        if (coeff.info(info_flags::real) && coeff < 0)
                        {
                            ex new_arg = -arg; // Flip argument to positive (e.g., -2*x -> 2*x)

                            // Apply the parity rules recursively
                            if (name == "cos")
                            {
                                // Even function: cos(-x) = cos(x)
                                return cos(new_arg).map(*this);
                            }
                            else if (name == "sin")
                            {
                                // Odd function: sin(-x) = -sin(x)
                                return -sin(new_arg).map(*this);
                            }
                            else if (name == "tan")
                            {
                                // Odd function: tan(-x) = -tan(x)
                                return -tan(new_arg).map(*this);
                            }
                        }
                    }
                }
                // Continue recursion for other parts of the expression
                return e.map(*this);
            }
        };

        ex current = e_in;
        parity_normalizer normalizer;

        // Step 1: Normalize arguments (handle negative signs).
        // Transforms -1/8*cos(-2*x) -> -1/8*cos(2*x).
        current = normalizer(current);

        // Step 2: Expand to resolve arithmetic signs and coefficients.
        // Handles double negatives, e.g., -1/2 * (-sin(x)) -> +1/2*sin(x).
        current = current.expand();

        // Step 3: Combine like terms.
        // Terms like -1/4*cos(2*x) and 1/4*cos(2*x) will cancel out here.
        current = current.normal();

        return current;
    }

    // --- TR10: Angle Contraction ---
    ex TR10_angle_contraction(const ex &e_in)
    {
        ex current = e_in;
        ex w = wild();

        // 1. Cosine double angle patterns
        current = current.subs(pow(cos(w), 2) - pow(sin(w), 2) == cos(2 * w), subs_options::algebraic);
        current = current.subs(2 * pow(cos(w), 2) - 1 == cos(2 * w), subs_options::algebraic);
        current = current.subs(1 - 2 * pow(sin(w), 2) == cos(2 * w), subs_options::algebraic);

        // 2. Sine double angle
        current = current.subs(sin(w) * cos(w) == numeric(1, 2) * sin(2 * w), subs_options::algebraic);

        return current;
    }

    // --- TR11: Angle Expansion ---
    // sin(2x) -> 2 sin(x) cos(x)
    ex TR11_angle_expansion(const ex &e_in)
    {
        struct double_angle_map : public map_function
        {
            ex operator()(const ex &e) override
            {
                if (is_a<function>(e))
                {
                    std::string name = ex_to<function>(e).get_name();
                    ex arg = e.op(0);
                    if (is_a<mul>(arg))
                    {
                        numeric n = arg.integer_content();
                        if (n.is_even())
                        {
                            ex half = arg / 2;
                            if (name == "sin")
                                return 2 * sin(half) * cos(half);
                            if (name == "cos")
                                return pow(cos(half), 2) - pow(sin(half), 2);
                        }
                    }
                }
                return e.map(*this);
            }
        };
        double_angle_map mapper;
        return mapper(e_in);
    }

    // --- New/Revised Strategies for a unified search ---

    // This strategy converts trig to exp.
    ex TR_to_exponential(const ex &e_in)
    {
        struct to_exp_map : public map_function
        {
            ex operator()(const ex &e) override
            {
                if (is_a<function>(e))
                {
                    std::string n = ex_to<function>(e).get_name();
                    ex x = e.op(0);
                    if (n == "sin")
                        return (exp(I * x) - exp(-I * x)) / (2 * I);
                    if (n == "cos")
                        return (exp(I * x) + exp(-I * x)) / 2;
                }
                return e.map(*this);
            }
        };
        to_exp_map mapper;
        return mapper(e_in);
    }

#if FALSE
    // Converter: Transforms pure exp(I*x) into cos(x) + I*sin(x).
    struct exp_to_trig_converter : public map_function
    {
        ex operator()(const ex &e) override
        {
            if (is_a<function>(e) && ex_to<function>(e).get_name() == "exp")
            {
                ex arg = e.op(0);
                if (arg.has(I))
                {
                    // Normalize arg to separate I.
                    // e.g., arg = I*x -> real_part = x
                    ex real_part = (arg / I).normal();
                    if (!real_part.has(I))
                    {
                        return cos(real_part) + I * sin(real_part);
                    }
                }
            }
            return e.map(*this);
        }
    };
#else
    // transformation of exp(...) functions.
    // It expands the argument, splits additive terms, and processes them individually:
    // 1. If a term has no 'I', keep as exp(term).
    // 2. If term/I has no 'I', convert to cos(x) + I*sin(x).
    // 3. Otherwise, recurse on the term.
    struct exp_to_trig_converter : public map_function
    {
        ex operator()(const ex &e) override
        {
            // Check if the expression is a function named "exp"
            if (is_a<function>(e) && ex_to<function>(e).get_name() == "exp")
            {
                ex arg = e.op(0);

                // Step 1: Expand the argument to handle internal parentheses,
                // subtractions, and distributions (e.g., I*(x+y) -> I*x + I*y).
                ex expanded_arg = arg.expand();

                // Prepare a container for the product of processed terms
                // Since exp(A+B) = exp(A) * exp(B), we multiply results.
                ex prod_result = 1;

                // Helper lambda to process each additive term according to the rules
                auto process_term = [&](const ex &term) -> ex
                {
                    // Rule 1: If the term does not contain the imaginary unit I
                    // Return exp(term) directly.
                    if (!term.has(I))
                    {
                        return exp(term);
                    }

                    // Calculate term / I to check if it's a pure imaginary phase
                    // We use expand() to simplify fractions like (I*x)/I -> x.
                    ex coeff = (term / I).expand();

                    // Rule 2: If (term / I) does not contain I
                    // It means term = I * real_val. Convert to Euler's formula.
                    if (!coeff.has(I))
                    {
                        return cos(coeff) + I * sin(coeff);
                    }

                    // Rule 3: If (term / I) still contains I
                    // The term is mixed or complex. Recursively map this converter
                    // onto the term inside a new exp(), ensuring sub-expressions
                    // (like nested exps) are processed.
                    return exp(term.map(*this));
                };

                // Step 2: Split the expanded argument into additive items
                if (is_a<add>(expanded_arg))
                {
                    // Iterate through all operands of the sum
                    for (size_t i = 0; i < expanded_arg.nops(); ++i)
                    {
                        prod_result *= process_term(expanded_arg.op(i));
                    }
                }
                else
                {
                    // If it's not a sum (just a single term), process it directly
                    prod_result *= process_term(expanded_arg);
                }

                return prod_result;
            }

            // Apply recursively to children of other functions/expressions
            return e.map(*this);
        }
    };
#endif

    struct exp_recu : public map_function
    {
        ex operator()(const ex &e) override
        {
            ex w1 = wild(1), w2 = wild(2);

            // 1. Bottom-Up Recursion
            // Apply map first to process children (operands) before the current node.
            ex c = e.map(*this);

            // 2. Optimization for Add/Subtract
            // The patterns exp(A)*exp(B) or exp(A)^B do not exist at the top level of an Add object.
            // Since children are already simplified by step 1, we can skip processing Add nodes entirely.
            if (is_a<add>(c))
            {
                return c;
            }

            // 3. Power Object Handling
            // We avoid calling expand() on Power objects to prevent massive polynomial expansion
            // (e.g., (A+B)^100). We only apply substitution to reduce exp(A)^B -> exp(A*B).
            if (is_a<power>(c))
            {
                return c.subs(pow(exp(w1), w2) == exp(w1 * w2), subs_options::algebraic);
            }

            // 4. General Case (Multiplication, Functions, etc.)
            // For Mul objects, expand() is necessary to distribute terms (e.g., (1+exp(x))*exp(-x) -> exp(-x) + 1).
            // Then apply substitutions to merge exponentials.
            return c.expand()
                .subs(pow(exp(w1), w2) == exp(w1 * w2), subs_options::algebraic)
                .subs(exp(w1) * exp(w2) == exp(w1 + w2), subs_options::algebraic);
        }
    };

    ex TR_from_exponential(const ex &e_in)
    {
        ex current = e_in;
        ex w1 = wild(1), w2 = wild(2);

        // std::ostringstream oss;
        // oss << "Transform input: " << current << "\n";

        // ============================================================
        // Step 1: Initial Cleanup
        // ============================================================
        // Force power expansion first to handle input like exp(-I*x)^(-1).
        // current = current.subs(pow(exp(w1), w2) == exp(w1 * w2));

        // ============================================================
        // Step 2: Iterative Normalization to Polynomials of exp(I*x)
        // ============================================================
        // We use a loop to repeatedly merge products and then split integers.
        // This ensures complex nested terms settle into a canonical rational form.
        exp_recu et;
        for (int i = 0; i < 10; ++i)
        {
            ex base = current;
            // 2.1 Merge exponential products: exp(A) * exp(B) -> exp(A+B)
            // current = current.subs(exp(w1) * exp(w2) == exp(w1 + w2), subs_options::algebraic);
            current = et(current);
            // oss << "Iter T0 " << i << ": " << current << "\n";
            current = current.expand();
            // oss << "Iter T1 " << i << ": " << current << "\n";

            if (current.is_equal(base))
            {
                break;
            }
        }

        // ============================================================
        // Step 3: Euler Transformation
        // ============================================================
        // Convert the unified base exp(I*x) to cos(x) + I*sin(x).
        exp_to_trig_converter converter;
        for (int i = 0; i < 5; ++i)
        {
            auto base = current;
            current = converter(current);
            current = minus_angle(current);
            current = current.normal();
            if (base.is_equal(current))
            {
                break;
            }
        }

        // ============================================================
        // Step 4: Trigonometric Simplification
        // ============================================================

        // Expand to handle the new trig expressions.
        current = current.expand();

        // Normalization is key here.
        // If we have (cos+Isin)^-1, normal() will rationalize the denominator:
        // 1/(c+Is) -> (c-Is)/(c^2+s^2).
        // GiNaC automatically simplifies c^2+s^2 to 1 for basic cases.
        current = current.normal();

        // oss << "Final Result: " << current << "\n";
        // emscripten_out(oss.str().c_str());
        return current;
    }

    // --- TR Inverse: Restore Tan/Sec ---
    ex TR_inverse_lookup(const ex &e_in)
    {
        ex current = e_in;
        ex w = wild();
        current = current.subs(sin(w) / cos(w) == tan(w), subs_options::algebraic);
        current = current.subs(cos(w) / sin(w) == pow(tan(w), -1), subs_options::algebraic);
        current = current.subs(pow(sin(w), 2) / pow(cos(w), 2) == pow(tan(w), 2), subs_options::algebraic);
        return current;
    }

    // =========================================================================
    // SECTION 3: The Driver (Beam Search / Population Based)
    // =========================================================================

    /**
     * Structure to hold candidate expressions with their scores.
     */
    struct Candidate
    {
        ex expression;
        long score;

        bool operator<(const Candidate &other) const
        {
            if (score != other.score)
                return score < other.score;
            return expression.compare(other.expression) < 0;
        }
    };

    /**
     * Apply all available algebraic strategies to a single expression.
     * Returns a list of new candidate expressions.
     */
    std::vector<ex> apply_all_strategies(const ex &e)
    {
        std::vector<ex> results;

        // 1. Basic Ops
        results.push_back(e.expand());
        results.push_back(e.normal());

        // 2. Pythagorean (TR5)
        results.push_back(TR5_pythagoras(e));

        // 3. Angle Contraction (TR10)
        results.push_back(TR10_angle_contraction(e));

        // 4. Angle Expansion (TR11)
        results.push_back(TR11_angle_expansion(e));

        // 5. Product to Sum (TR8)
        results.push_back(TR8_product_to_sum(e));

        // 6. Trig Sum Expansion (TR2)
        results.push_back(TR2_expand_trig_sums(e));

        // 7. Sum to Product (TR9)
        results.push_back(TR9_sum_to_product(e));

        // 8. Power Reduction (TR7)
        results.push_back(TR7_power_reduction(e));

        return results;
    }

    ex fu_full_simplification(const ex &e_in)
    {
        // Beam Search Parameters
        const int BEAM_WIDTH = 5;
        const double SCORE_THRESHOLD = 2.0;
        const int MAX_DEPTH = 5;

        std::set<ex, ex_is_less> visited;
        std::vector<Candidate> population;

        // Initialize
        ex current_best_ex = e_in;
        long current_min_score = get_complexity(e_in);
        population.push_back({e_in, current_min_score});

        // push minus angle
        ex tmp = minus_angle(e_in);
        long tmp_score = get_complexity(tmp);
        population.push_back({tmp, tmp_score});

        // std::ostringstream oss;
        // oss << e_in << "\nminus angle : " << tmp << "\n";
        // emscripten_out(oss.str().c_str());

        visited.insert(e_in);

        for (int depth = 0; depth < MAX_DEPTH; ++depth)
        {
            std::vector<Candidate> next_gen;

            // Generate children from current population
            for (const auto &parent : population)
            {
                std::vector<ex> children = apply_all_strategies(parent.expression);

                for (const auto &child : children)
                {
                    if (visited.count(child))
                        continue;
                    visited.insert(child);

                    ex normalized_child = minus_angle(child.normal());
                    long score = get_complexity(normalized_child);

                    // Update global best
                    if (score < current_min_score)
                    {
                        current_min_score = score;
                        current_best_ex = normalized_child;
                    }

                    next_gen.push_back({normalized_child, score});
                }
            }

            if (next_gen.empty())
                break;

            // Sort by score
            std::sort(next_gen.begin(), next_gen.end());

            // Selection Logic (Beam Filter)
            population.clear();
            long limit_score = static_cast<long>(current_min_score * SCORE_THRESHOLD);

            int count = 0;
            for (const auto &cand : next_gen)
            {
                if (count < BEAM_WIDTH || cand.score <= limit_score)
                {
                    population.push_back(cand);
                    count++;
                }
                else
                {
                    break;
                }
            }
        }

        return current_best_ex;
    }

    struct map_partial : public map_function
    {
        ex operator()(const ex &e) override
        {
            // Check if the expression is a function named "exp"
            if (is_a<function>(e))
            {
                function f = ex_to<function>(e);
                if (f.get_name() != "sin" && f.get_name() != "cos" && f.get_name() != "sinh" && f.get_name() != "cosh" && f.get_name() != "pow" && f.get_name() != "exp" && f.nops() == 1)
                {
                    ex arg = e.op(0);
                    arg = TR_from_exponential(arg);
                    arg = fu_full_simplification(arg);
                    return function(f.get_serial(), arg.map(*this));
                }
            }
            return e.map(*this);
        }
    };

    /**
     * Main Entry Point: trigsimp (New SOTA-inspired driver)
     */
    ex trigsimp(const ex &expression)
    {
        if (expression.info(info_flags::numeric))
            return expression;

        // 1. Initial Normalization
        ex initial_form = TR1_normalize(expression).normal();

        // 2. Run the main simplification engine, which now includes all strategies.
        ex best_result = fu_full_simplification(initial_form);

        ex toexp = TR_to_exponential(initial_form);
        {
            map_partial mp;
            ex pm = mp(toexp);
            if (get_complexity(pm) < get_complexity(best_result))
            {
                best_result = pm;
            }
        }

        toexp = TR_from_exponential(toexp);
        toexp = fu_full_simplification(toexp);
        if (get_complexity(toexp) < get_complexity(best_result))
        {
            best_result = toexp;
        }

        // 3. Final polish and return
        ex res_inv = TR_inverse_lookup(best_result);
        if (get_complexity(res_inv) < get_complexity(best_result))
        {
            best_result = res_inv;
        }

        return best_result.normal();
    }

} // namespace GinacTrig