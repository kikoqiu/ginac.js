const createGinacModule = require('../dist/ginac.js');
const ginac = require('../dist/index.umd.js');

// Increase timeout for WebAssembly initialization and complex calculations
jest.setTimeout(20000);

describe('GiNaC Integration Tests', () => {

    // Initialize the WASM module before running tests
    beforeAll(async () => {
        let module = await createGinacModule();
        await ginac.initGiNaC(module);
    });

    /**
     * Category 1: Basic Polynomials & Powers
     * Testing rule: ∫ x^n dx = x^(n+1) / (n+1)
     */
    describe('Polynomials & Power Functions', () => {
        
        // 1. Constant
        test('Constant: 5', () => {
            const val = ginac.integ("5", "x");
            expect(ginac.op_subtract(val,"5*x").toString()).toEqual("0");
        });

        // 2. Linear x
        test('Linear: x', () => {
            const val = ginac.integ("x", "x");
            expect(ginac.op_subtract(val,"1/2*x^2").toString()).toEqual("0");
        });

        // 3. Simple Power x^2
        test('Power: x^2', () => {
            const val = ginac.integ("pow(x,2)", "x");
            expect(ginac.op_subtract(val,"1/3*x^3").toString()).toEqual("0");
        });

        // 4. Higher Power x^3
        test('Power: x^3', () => {
            const val = ginac.integ("pow(x,3)", "x");
            expect(ginac.op_subtract(val,"1/4*x^4").toString()).toEqual("0");
        });

        // 5. Negative Power (1/x^2)
        test('Negative Power: x^-2', () => {
            const val = ginac.integ("pow(x,-2)", "x");
            // Expected: -1/x
            expect(ginac.op_subtract(val,"-x^(-1)").toString()).toEqual("0"); 
        });

        // 6. Fractional Power (Square root)
        test('Root: sqrt(x)', () => {
            const val = ginac.integ("sqrt(x)", "x");
            // Expected: 2/3 * x^(3/2)
            expect(ginac.op_subtract(val,"2/3*x^(3/2)").toString()).toEqual("0");
        });

        // 7. Polynomial Sum: x^2 + x
        test('Sum: x^2 + x', () => {
            const val = ginac.integ("x^2 + x", "x");
            expect(ginac.op_subtract(val,"1/2*x^2+1/3*x^3").toString()).toEqual("0");
        });

        // 8. Polynomial with coefficients
        test('Coefficients: 3*x^2 - 2*x', () => {
            const val = ginac.integ("3*x^2 - 2*x", "x");
            expect(ginac.op_subtract(val,"-x^2+x^3").toString()).toEqual("0");
        });

        // 9. Expansion: (x+1)^2
        test('Expansion: (x+1)^2', () => {
            // (x+1)^2 = x^2 + 2x + 1 -> x^3/3 + x^2 + x
            const val = ginac.integ("pow(x+1, 2)", "x");
            expect(ginac.op_subtract(val,"x^2+x+1/3*x^3").toString()).toEqual("0");
        });
    });

    /**
     * Category 2: Trigonometric Functions
     * Testing sin, cos, tan, and basic identities
     */
    describe('Trigonometric Functions', () => {

        // 10. sin(x)
        test('sin(x)', () => {
            const val = ginac.integ("sin(x)", "x");
            expect(ginac.op_subtract(val,"-cos(x)").toString()).toEqual("0");
        });

        // 11. cos(x)
        test('cos(x)', () => {
            const val = ginac.integ("cos(x)", "x");
            expect(ginac.op_subtract(val,"sin(x)").toString()).toEqual("0");
        });

        // 12. tan(x) -> -ln|cos(x)|
        test('tan(x)', () => {
            const val = ginac.integ("tan(x)", "x");
            expect(ginac.op_subtract(val,"-log(cos(x))").toString()).toEqual("0");
        });

        // 13. cot(x) -> ln|sin(x)|
        test('cot(x)', () => {
            const val = ginac.integ("1/tan(x)", "x");
            expect(ginac.op_subtract(val,"log(sin(x))").toString()).toEqual("0");
        });

        // 14. sin(2x) -> Chain rule
        test('sin(2*x)', () => {
            const val = ginac.integ("sin(2*x)", "x");
            expect(ginac.op_subtract(val,"-1/2*cos(2*x)").toString()).toEqual("0");
        });

        // 15. cos(3x)
        test('cos(3*x)', () => {
            const val = ginac.integ("cos(3*x)", "x");
            expect(ginac.op_subtract(val,"1/3*sin(3*x)").toString()).toEqual("0");
        });

        // 16. sec^2(x) -> tan(x) (Input as 1/cos^2)
        test('sec(x)^2', () => {
            const val = ginac.integ("1/pow(cos(x),2)", "x");
            expect(ginac.op_subtract(val,"tan(x)").toString()).toEqual("0");
        });

        // 17. csc^2(x) -> -cot(x) (Input as 1/sin^2)
        test('csc(x)^2', () => {
            const val = ginac.integ("1/pow(sin(x),2)", "x");
            // Output might vary slightly based on GiNaC version, usually -cot(x)
            expect(ginac.op_subtract(val,"-1/tan(x)").toString()).toEqual("0");
        });

        // 18. sin(x)*cos(x) -> -1/4*cos(2x) or 1/2*sin(x)^2
        test('sin(x)*cos(x)', () => {
            const val = ginac.integ("sin(x)*cos(x)", "x");
            expect(ginac.op_subtract(val,"-1/4*cos(2*x)").toString()).toEqual("0");
        });

        // 19. cos^2(x) -> x/2 + 1/4*sin(2x)
        test('cos(x)^2', () => {
            const val = ginac.integ("pow(cos(x),2)", "x");
            expect(ginac.op_subtract(val,"1/4*sin(2*x)+1/2*x").toString()).toEqual("0");
        });
    });

    /**
     * Category 3: Exponential & Logarithmic Functions
     * Testing e^x, log(x), a^x
     */
    describe('Exponential & Logarithmic', () => {

        // 20. e^x
        test('exp(x)', () => {
            const val = ginac.integ("exp(x)", "x");
            expect(ginac.op_subtract(val,"exp(x)").toString()).toEqual("0");
        });

        // 21. e^(2x)
        test('exp(2*x)', () => {
            const val = ginac.integ("exp(2*x)", "x");
            expect(ginac.op_subtract(val,"1/2*exp(2*x)").toString()).toEqual("0");
        });

        // 22. e^(-x)
        test('exp(-x)', () => {
            const val = ginac.integ("exp(-x)", "x");
            expect(ginac.op_subtract(val,"-exp(-x)").toString()).toEqual("0");
        });

        // 23. 1/x -> log(x)
        test('Reciprocal: 1/x', () => {
            const val = ginac.integ("1/x", "x");
            expect(ginac.op_subtract(val,"log(x)").toString()).toEqual("0");
        });

        // 24. 1/(x+1)
        test('Shifted Reciprocal: 1/(x+1)', () => {
            const val = ginac.integ("1/(x+1)", "x");
            expect(ginac.op_subtract(val,"log(x+1)").toString()).toEqual("0");
        });

        // 25. log(x) (Integration by parts)
        test('log(x)', () => {
            const val = ginac.integ("log(x)", "x");
            // x*log(x) - x
            expect(ginac.op_subtract(val,"x*log(x)-x").toString()).toEqual("0");
        });

        // 26. 2^x -> 2^x / ln(2)
        test('Base 2 Exponential: 2^x', () => {
            const val = ginac.integ("pow(2,x)", "x");
            // GiNaC might output log(2)^(-1)
            expect(ginac.op_subtract(val,"2^x*log(2)^(-1)").toString()).toEqual("0");
        });

        // 27. x * e^x (Integration by parts)
        test('x*exp(x)', () => {
            const val = ginac.integ("x*exp(x)", "x");
            // (x-1)*e^x -> x*exp(x) - exp(x)
            expect(ginac.op_subtract(val,"x*exp(x)-exp(x)").toString()).toEqual("0");
        });

        // 28. x * log(x)
        test('x*log(x)', () => {
            const val = ginac.integ("x*log(x)", "x");
            // x^2/2 * log(x) - x^2/4
            expect(ginac.op_subtract(val,"1/2*x^2*log(x)-1/4*x^2").toString()).toEqual("0");
        });
    });

    /**
     * Category 4: Rational Functions & Inverse Trigonometry
     * Testing atan, asin, and rational decomposition
     */
    describe('Rational & Inverse Trig', () => {

        // 29. 1/(1+x^2) -> atan(x)
        test('Arctan derivative: 1/(1+x^2)', () => {
            const val = ginac.integ("1/(1+x^2)", "x");
            expect(ginac.op_subtract(val,"atan(x)").toString()).toEqual("0");
        });

        // 30. 1/sqrt(1-x^2) -> asin(x)
        test('Arcsin derivative: 1/sqrt(1-x^2)', () => {
            const val = ginac.integ("1/sqrt(1-x^2)", "x");
            expect(ginac.op_subtract(val,"asin(x)").toString()).toEqual("0");
        });

        // 31. 1/(4+x^2) -> 1/2*atan(x/2)
        test('Scaled Arctan: 1/(4+x^2)', () => {
            const val = ginac.integ("1/(4+x^2)", "x");
            expect(ginac.op_subtract(val,"1/2*atan(1/2*x)").toString()).toEqual("0");
        });

        // 32. x/(1+x^2) -> 1/2*ln(1+x^2) (u-substitution)
        test('Substitution: x/(1+x^2)', () => {
            const val = ginac.integ("x/(1+x^2)", "x");
            expect(ginac.op_subtract(val,"1/2*log(x^2+1)").toString()).toEqual("0");
        });

        // 33. -1/sqrt(1-x^2) -> acos(x) or -asin(x)
        test('Arccos derivative check', () => {
            const val = ginac.integ("-1/sqrt(1-x^2)", "x");
            // GiNaC usually prefers -asin(x) for simplicity
            expect(ginac.op_subtract(val,"-asin(x)").toString()).toEqual("0");
        });

        // 34. Inverse sine integral: asin(x)
        test('Integral of asin(x)', () => {
            const val = ginac.integ("asin(x)", "x");
            // x*asin(x) + sqrt(1-x^2)
            expect(ginac.op_subtract(val,"x*asin(x)+sqrt(-x^2+1)").toString()).toEqual("0");
        });

        // 35. Inverse tangent integral: atan(x)
        test('Integral of atan(x)', () => {
            const val = ginac.integ("atan(x)", "x");
            // x*atan(x) - 1/2*ln(1+x^2)
            expect(ginac.op_subtract(val,"x*atan(x)-1/2*log(x^2+1)").toString()).toEqual("0");
        });
    });

    /**
     * Category 5: Hyperbolic Functions
     * Testing sinh, cosh, tanh
     */
    describe('Hyperbolic Functions', () => {

        // 36. sinh(x)
        test('sinh(x)', () => {
            const val = ginac.integ("sinh(x)", "x");
            expect(ginac.op_subtract(val,"cosh(x)").toString()).toEqual("0");
        });

        // 37. cosh(x)
        test('cosh(x)', () => {
            const val = ginac.integ("cosh(x)", "x");
            expect(ginac.op_subtract(val,"sinh(x)").toString()).toEqual("0");
        });

        // 38. tanh(x)
        test('tanh(x)', () => {
            const val = ginac.integ("tanh(x)", "x");
            expect(ginac.op_subtract(val,"log(cosh(x))").toString()).toEqual("0");
        });

        // 39. sinh(ax)
        test('sinh(2*x)', () => {
            const val = ginac.integ("sinh(2*x)", "x");
            expect(ginac.op_subtract(val,"1/2*cosh(2*x)").toString()).toEqual("0");
        });

        // 40. sech^2(x) -> tanh(x)
        test('sech(x)^2', () => {
            const val = ginac.integ("1/pow(cosh(x),2)", "x");
            expect(ginac.op_subtract(val,"tanh(x)").toString()).toEqual("0");
        });
    });

    /**
     * Category 6: Symbolic Constants & Mixed Integration
     * Testing integration with multiple variables (treating others as constants)
     */
    describe('Symbolic & Mixed Cases', () => {

        // 41. Constant 'a' -> a*x
        test('Symbolic Constant: a', () => {
            const val = ginac.integ("a", "x");
            expect(ginac.op_subtract(val,"a*x").toString()).toEqual("0");
        });

        // 42. a*x^2
        test('Symbolic Coefficient: a*x^2', () => {
            const val = ginac.integ("a*x^2", "x");
            expect(ginac.op_subtract(val,"1/3*a*x^3").toString()).toEqual("0");
        });

        // 43. sin(a*x)
        test('Symbolic Frequency: sin(a*x)', () => {
            const val = ginac.integ("sin(a*x)", "x");
            // -1/a * cos(a*x)
            expect(ginac.op_subtract(val,"-cos(a*x)*a^(-1)").toString()).toEqual("0");
        });

        // 44. exp(a*x)
        test('Symbolic Exponential: exp(a*x)', () => {
            const val = ginac.integ("exp(a*x)", "x");
            expect(ginac.op_subtract(val,"exp(a*x)*a^(-1)").toString()).toEqual("0");
        });

        // 45. x + y (Integrate wrt x)
        test('Multi-variable Sum: x + y', () => {
            const val = ginac.integ("x+y", "x");
            expect(ginac.op_subtract(val,"1/2*x^2+x*y").toString()).toEqual("0");
        });

        // 46. 1/(x+a)
        test('Symbolic Logarithm: 1/(x+a)', () => {
            const val = ginac.integ("1/(x+a)", "x");
            expect(ginac.op_subtract(val,"log(x+a)").toString()).toEqual("0");
        });

        // 47. x*sin(x) -> Parts: sin(x) - x*cos(x)
        test('Parts: x*sin(x)', () => {
            const val = ginac.integ("x*sin(x)", "x");
            expect(ginac.op_subtract(val,"sin(x)-x*cos(x)").toString()).toEqual("0");
        });

        // 48. x*cos(x) -> Parts: cos(x) + x*sin(x)
        test('Parts: x*cos(x)', () => {
            const val = ginac.integ("x*cos(x)", "x");
            expect(ginac.op_subtract(val,"cos(x)+x*sin(x)").toString()).toEqual("0");
        });

        // 49. (x^2+1)/x -> x + 1/x -> x^2/2 + log(x)
        test('Rational Split: (x^2+1)/x', () => {
            const val = ginac.integ("(x^2+1)/x", "x");
            expect(ginac.op_subtract(val,"1/2*x^2+log(x)").toString()).toEqual("0");
        });

        // 50. cos(x+t) -> Expansion formula result
        test('Angle Sum: cos(x+t)', () => {
            const val = ginac.integ("cos(x+t)", "x");
            // cos(x+t) integrates to sin(x+t). 
            // Depending on normalization, GiNaC might expand it to: sin(x+t)
            expect(ginac.op_subtract(val,"sin(x+t)").toString()).toEqual("0");
        });
    });
});




describe('base', () => {

    beforeEach(async () => {
        let module=await createGinacModule();
        await ginac.initGiNaC(module);
    });
    test('pow(x,3)+pow(x,2)+1', async () => {
        const val=ginac.integ("pow(x,3)+pow(x,2)+1","x");
        expect(ginac.op_subtract(val,"1/4*x^4+x+1/3*x^3").toString()).toEqual("0");
    });
    test('cos(x)*sin(x)', async () => {
        const val=ginac.integ("cos(x)*sin(x)","x");
        expect(ginac.op_subtract(val,"-1/4*cos(2*x)").toString()).toEqual("0");
    });
    test('tan', async () => {
        const val=ginac.integ("tan(x)","x");
        expect(ginac.op_subtract(val,"-log(cos(x))").toString()).toEqual("0");
    });
    test('cos(x)^2', async () => {
        const val=ginac.integ("cos(x)^2","x");
        expect(ginac.op_subtract(val,"1/4*sin(2*x)+1/2*x").toString()).toEqual("0");
    });
    test('cos(x+t)', async () => {
        const val=ginac.integ("cos(x+t)","x");
        expect(ginac.op_subtract(val,"sin(t+x)").toString()).toEqual("0");
    });

    test('pow(3,x)', async () => {
        const val=ginac.integ("pow(3,x)","x");
        expect(ginac.op_subtract(val,"3^x*log(3)^(-1)").toString()).toEqual("0");   
    });

    test('exp(sin(x))', async () => {
        const val=ginac.integ("exp(sin(x))","x");
        expect(val.toString()).toEqual("integral(x,0,x,exp(sin(x)))");
    });


});
