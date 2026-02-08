const createGinacModule = require('../dist/ginac.js');
const ginac = require('../dist/index.umd.js');
jest.setTimeout(1000);

const trimap = [
  //  (Pythagorean Identity)
    ["sin(x)^2 + cos(x)^2", "1"],
    ["1 - sin(x)^2", "cos(x)^2"],

    //  (Hyperbolic Identity)
    ["cosh(x)^2 - sinh(x)^2", "1"],

    // (Basic Definition)
    ["tan(x) - sin(x)/cos(x)", "0"],
    ["1/tan(x)*sin(x)", "cos(x)"],

    // (Double Angle Contraction - TR10)
    ["2*sin(x)*cos(x)", "sin(2*x)"],
    ["cos(x)^2 - sin(x)^2", "cos(2*x)"],
    ["1 - 2*sin(x)^2", "cos(2*x)"],

    // 5. (Power Reduction - TR7)
    // sin^4 - cos^4 = (sin^2 - cos^2)(sin^2 + cos^2) = -cos(2x) * 1
    ["sin(x)^4 - cos(x)^4", "-cos(2*x)"],

    // 6.  (Expanded Square)
    // (sin x + cos x)^2 = 1 + 2sin x cos x = 1 + sin 2x
    ["(sin(x) + cos(x))^2", "1+sin(2*x)"],

    // 7.  (Product-to-Sum - TR8)
    // cos(x+y) = cosx cosy - sinx siny => cosx cosy
    ["cos(x+y) + sin(x)*sin(y)", "cos(y)*cos(x)"],

    // 8. (Rational Expression - TR12 Exponential mostly)
    // (1 - cos 2x) = 2 sin^2 x => 除以 2 sin x => sin x
    ["(1 - cos(2*x)) / (2*sin(x))", "sin(x)"],

    // 9.  (Reciprocal)
    ["sin(x) * (1/sin(x))", "1"],

    // 10. mixed
    ["sin(x)^2 + cos(x)^2 + sin(y)^2 + cos(y)^2", "2"]
];

describe('base', () => {

    beforeEach(async () => {
        let module=await createGinacModule();
        await ginac.initGiNaC(module);
    });
   
    test('trigsimp', async () => {
        trimap.forEach(
            a=>{
                let v=ginac.trigsimp(a[0]).toString();
                expect(ginac.trigsimp(a[0]).toString()).toEqual(a[1]);
            }
        );
    });
    

});
