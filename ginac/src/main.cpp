
#include <ginac/ginac.h>
#include <ginac/parser.h>
#include <emscripten.h>
#include <emscripten/console.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

#include "buildin.h"

using std::string;
using std::multimap;
using std::make_pair;
using namespace GiNaC;

// ============================================================================
// Global lookup map
// ============================================================================
// Table of functions (a multimap, because one function may appear with different
// numbers of parameters)
typedef ex (*fcnp)(const exprseq &e);
typedef ex (*fcnp2)(const exprseq &e, int serial);

struct fcn_desc {
	fcn_desc() : p(nullptr), num_params(0), is_ginac(false), serial(0) {}
	fcn_desc(fcnp func, int num) : p(func), num_params(num), is_ginac(false), serial(0) {}
	fcn_desc(fcnp2 func, int num, int ser) : p((fcnp)func), num_params(num), is_ginac(true), serial(ser) {}

	fcnp p;		// Pointer to function
	int num_params;	// Number of parameters (0 = arbitrary)
	bool is_ginac;	// Flag: function is GiNaC function
	int serial;	// GiNaC function serial number (if is_ginac == true)
};

typedef multimap<string, fcn_desc> fcn_tab;
static fcn_tab fcns;


static void insert_fcns(const fcn_init *p)
{
	while (p->name) {
		fcns.insert(make_pair(string(p->name), fcn_desc(p->p, p->num_params)));
		p++;
	}
}

static ex f_ginac_function(const exprseq &es, int serial)
{
	return GiNaC::function(serial, es);
}



namespace GiNaC {
static void ginsh_get_ginac_functions(void)
{
	unsigned serial = 0;
	for (auto & i : GiNaC::function::get_registered_functions()) {
		fcns.insert(make_pair(i.get_name(), fcn_desc(f_ginac_function, i.get_nparams(), serial)));
		serial++;
	}
}
}


static void init() {
    if (!fcns.empty()) return;
    insert_fcns(builtin_fcns);
    ginsh_get_ginac_functions();
}

// ============================================================================
// Output Manager & Protocol Helpers
// ============================================================================

// Structure matching JS definitions for return values
struct ReturnBox {
    int32_t type;   // 0=Error, 1=Success(String/Archive), >1=Special
    int32_t len;    // Length of data
    const char* ptr;// Pointer to data
};

// Singleton Output Manager to handle return memory safely
class OutputManager {
public:
    std::ostringstream oss;
    
private:
    std::string _cache; // Keeps the data alive after oss.str() is called
    ReturnBox box;

public:
    OutputManager() {
        oss.sync_with_stdio(false);
    }

    // Reset the stream for new operation
    void reset() {
        oss.str("");
        oss.clear();
    }

    #if __cplusplus >= 202002L
        ReturnBox* commit(int type) {        
            // C++20: Zero-copy access to internal buffer
            auto sv = oss.view();
            box.type = type;
            box.ptr = sv.data();
            box.len = static_cast<int32_t>(sv.size());
            return &box;
        }
    #else
        // Commit current oss content to the return box
        ReturnBox* commit(int type) {
            _cache = oss.str(); // Copy from stream to string to own the memory
            box.type = type;
            box.len = static_cast<int32_t>(_cache.size());
            box.ptr = _cache.c_str();
            return &box;
        }
    #endif



    // Helper for error reporting
    ReturnBox* error(const char* msg) {
        reset();
        oss << msg;
        return commit(0);
    }
};

symtab lst_to_symtab(const lst & l) {
    symtab table;
    for (auto & item : l) {
        // Ensure the item is actually a symbol before adding
        if (is_a<symbol>(item)) {
            const symbol & s = ex_to<symbol>(item);
            table[s.get_name()] = s;
        }
    }
    return table;
}

static OutputManager g_out;

// Helper: Deserialize JS Input (String or Uint8Array/Archive) -> ex
ex raw_to_ex(const char* data, int len, int type, const lst &symlist) {
    // type: 0 = string (equation), 1 = binary (archive), 2 = binary number (double)
    if (type == 0) {
        std::string s(data, len);
        // Quick check: if it happens to be a raw binary string starting with GBARY
        if (s.size() > 5 && s.substr(0, 5) == "GBARY") {
             std::istringstream iss(s);
             archive ar;
             iss >> ar;
             return ar.unarchive_ex(symlist);
        }            
        symtab m = lst_to_symtab(symlist);
        parser p(m);
        return p(s);        
    } else if(type == 1) {
        std::string bin_str(data, len);
        std::istringstream iss(bin_str);
        archive ar;
        iss >> ar;
        return ar.unarchive_ex(symlist);
    } else if(type==2){
        if(len != sizeof(double)) throw std::runtime_error("Invalid binary number length");
        double val;
        memcpy(&val, data, sizeof(double));
        return GiNaC::numeric(val);
    } else if(type==3){// symbolic number
        std::string s(data, len);        
        return GiNaC::numeric(s.c_str());
    } else if(type==4){// symbol
        std::string s(data, len);
        for (auto & item : symlist) {
            if (is_a<symbol>(item) && ex_to<symbol>(item).get_name() == s) {
                return item;
            }
        }
        return GiNaC::symbol(s);
    } else {
        throw std::runtime_error("Unknown input type");
    }
}



// ============================================================================
// Exported Functions (Manual Binding)
// ============================================================================

extern "C" {
// Returns a JSON list of all exported functions
EMSCRIPTEN_KEEPALIVE
ReturnBox* raw_get_all_exported_functions() {
    init();
    g_out.reset();
    
    g_out.oss << "[";
    bool first = true;
    for (const auto & pair : fcns) {
        if (!first) g_out.oss << ",";
        first = false;
        g_out.oss << "{\"name\":\"" << pair.first << "\",\"nparams\":" << pair.second.num_params << "}";
    }
    g_out.oss << "]";
    
    return g_out.commit(1); // Type 1 = JSON String
}

// Main execution function
// Protocol: args_buffer = [NumArgs(4B)] + ([Type(1B)][Len(4B)][Data...])*n
EMSCRIPTEN_KEEPALIVE
ReturnBox* raw_callFunc(const char* funcName, const char* args_buffer) {
    init();
    g_out.reset();

    try {
        std::string sFuncName(funcName);
        
        // Unpack arguments count
        int num_args;
        memcpy(&num_args, args_buffer, 4);
        int offset = 4;

        // Function lookup logic (same as original)
        fcn_tab::iterator var_func=fcns.end(), fixed_func=fcns.end();
        for(auto it = fcns.lower_bound(sFuncName); it != fcns.upper_bound(sFuncName); ++it) {
            if (it->second.num_params == -1){
                var_func = it;
                continue;
            }
            if(it->second.num_params == num_args) {
                fixed_func = it;
                break;
            }
        }
        
        fcn_desc func_info;
        if (fixed_func != fcns.end()) {
            func_info = fixed_func->second;
        } else if (var_func != fcns.end()) {
            func_info = var_func->second;
        } else {
            return g_out.error(("Function not found: " + sFuncName).c_str());
        }

        if (func_info.num_params != -1 && num_args != func_info.num_params) {
            return g_out.error("Incorrect number of arguments");
        }

        // Parse arguments        
        exprseq es;
        GiNaC::exset syms;
        for (int i = 0; i < num_args; ++i) {
            uint8_t type = args_buffer[offset];
            offset += 1;
            uint32_t len;
            memcpy(&len, args_buffer + offset, 4);
            offset += 4;

            lst symlist;            
            for (auto const &element : syms) {
                symlist.append(element);
            }
            auto ex_arg = raw_to_ex(args_buffer + offset, len, type, symlist);            
            find_all_symbols(ex_arg, syms);
            es.append(ex_arg);
            offset += len;
        }

        // Execute
        ex result_ex;
        if(!func_info.is_ginac){
            result_ex = func_info.p(es); 
        }else{
            result_ex = ((fcnp2)(func_info.p))(es,func_info.serial); 
        }
        

        // Pack result (Success, Binary Archive)
        archive ar;
        ar.archive_ex(result_ex, "");
        g_out.oss << ar;        
        return g_out.commit(1);

    } catch (std::exception &e) {
        return g_out.error(e.what());
    } catch (...) {
        return g_out.error("Unknown error occurred in callFunc");
    }
}

// Enum mapping for result types
enum ResultType {
    RES_STRING = 1,
    RES_JSON = 2,
    RES_NUMBER = 3,
    RES_MATRIX = 4,
    RES_LATEX = 5
};

// Convert Archive -> Desired Format
EMSCRIPTEN_KEEPALIVE
ReturnBox* raw_getResult(const char* data, int len, int input_type, int output_format) {
    g_out.reset();

    try {
        ex e = raw_to_ex(data, len, input_type, lst{});

        switch (output_format) {
            case RES_STRING: {
                g_out.oss << e;
                return g_out.commit(RES_STRING);
            }
            case RES_LATEX: {
                g_out.oss<< latex << e << dflt;
                return g_out.commit(RES_LATEX);
            }
            case RES_JSON: {
                // Manual JSON construction
                if (is_a<numeric>(e)) {
                    g_out.oss << "{\"type\":\"number\",\"value\":" << ex_to<numeric>(e).to_double() << "}";
                } else if (is_a<symbol>(e)) {
                    g_out.oss << "{\"type\":\"symbol\",\"name\":\"" << ex_to<symbol>(e).get_name() << "\"}";
                } else if (is_a<matrix>(e)) {
                    const matrix &m = ex_to<matrix>(e);
                    g_out.oss << "{\"type\":\"matrix\",\"rows\":" << m.rows() << ",\"cols\":" << m.cols() << "}";
                } else {
                    std::ostringstream tmp; tmp << e;
                    g_out.oss << "{\"type\":\"expression\",\"repr\":\"" << tmp.str() << "\"}";
                }
                return g_out.commit(RES_JSON);
            }
            case RES_NUMBER: {
                ex eval = e.evalf();
                if (is_a<numeric>(eval)) {
                    double val = ex_to<numeric>(eval).to_double();
                    g_out.oss.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
                // If not number, empty oss -> len 0 -> null in JS
                return g_out.commit(RES_NUMBER);
            }
            case RES_MATRIX: {
                int rows = 1, cols = 1;
                bool is_mat = is_a<matrix>(e);
                if (is_mat) {
                    const matrix &m = ex_to<matrix>(e);
                    rows = m.rows();
                    cols = m.cols();
                }

                // Binary Matrix Protocol: [Rows][Cols][CellData...]
                g_out.oss.write(reinterpret_cast<const char*>(&rows), 4);
                g_out.oss.write(reinterpret_cast<const char*>(&cols), 4);

                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        ex elem = is_mat ? ex_to<matrix>(e)(r, c).evalf() : e.evalf();
                        if (is_a<numeric>(elem)) {
                            // Type 1: Double
                            char t = 1; g_out.oss.write(&t, 1);
                            double d = ex_to<numeric>(elem).to_double();
                            g_out.oss.write(reinterpret_cast<const char*>(&d), 8);
                        } else {
                            // Type 2: String
                            char t = 2; g_out.oss.write(&t, 1);
                            std::ostringstream ss; ss << elem;
                            std::string s = ss.str();
                            int32_t sl = (int32_t)s.size();
                            g_out.oss.write(reinterpret_cast<const char*>(&sl), 4);
                            g_out.oss.write(s.c_str(), sl);
                        }
                    }
                }
                return g_out.commit(RES_MATRIX);
            }
        }
        return g_out.error("Unknown result type");
    } catch (std::exception &e) {
        return g_out.error(e.what());
    }
}

} // extern "C"