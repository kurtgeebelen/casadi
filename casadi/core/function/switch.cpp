/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include "switch.hpp"

using namespace std;

namespace casadi {

  Switch::Switch(const std::string& name,
                 const std::vector<Function>& f, const Function& f_def)
    : FunctionInternal(name), f_(f), f_def_(f_def) {

    // Consitency check
    casadi_assert(!f_.empty());
  }

  Switch::~Switch() {
  }

  size_t Switch::get_n_in() {
    for (auto&& i : f_) if (!i.is_null()) return 1+i.n_in();
    casadi_assert(!f_def_.is_null());
    return 1+f_def_.n_in();
  }

  size_t Switch::get_n_out() {
    for (auto&& i : f_) if (!i.is_null()) return i.n_out();
    casadi_assert(!f_def_.is_null());
    return f_def_.n_out();
  }

  Sparsity Switch::get_sparsity_in(int i) {
    if (i==0) {
      return Sparsity::scalar();
    } else {
      Sparsity ret;
      for (auto&& fk : f_) {
        if (!fk.is_null()) {
          const Sparsity& s = fk.sparsity_in(i-1);
          ret = ret.is_null() ? s : ret.unite(s);
        }
      }
      casadi_assert(!f_def_.is_null());
      const Sparsity& s = f_def_.sparsity_in(i-1);
      ret = ret.is_null() ? s : ret.unite(s);
      return ret;
    }
  }

  Sparsity Switch::get_sparsity_out(int i) {
    Sparsity ret;
    for (auto&& fk : f_) {
      if (!fk.is_null()) {
        const Sparsity& s = fk.sparsity_out(i);
        ret = ret.is_null() ? s : ret.unite(s);
      }
    }
    casadi_assert(!f_def_.is_null());
    const Sparsity& s = f_def_.sparsity_out(i);
    ret = ret.is_null() ? s : ret.unite(s);
    return ret;
  }

  void Switch::init(const Dict& opts) {
    // Call the initialization method of the base class
    FunctionInternal::init(opts);

    // Buffer for mismatching sparsities
    size_t sz_buf=0;

    // Get required work
    for (int k=0; k<=f_.size(); ++k) {
      const Function& fk = k<f_.size() ? f_[k] : f_def_;
      if (fk.is_null()) continue;

      // Memory for evaluation
      alloc(fk);

      // Required work vectors
      size_t sz_buf_k=0;

      // Add size for input buffers
      for (int i=1; i<n_in(); ++i) {
        const Sparsity& s = fk.sparsity_in(i-1);
        if (s!=sparsity_in(i)) {
          alloc_w(s.size1()); // for casadi_project
          sz_buf_k += s.nnz();
        }
      }

      // Add size for output buffers
      for (int i=0; i<n_out(); ++i) {
        const Sparsity& s = fk.sparsity_out(i);
        if (s!=sparsity_out(i)) {
          alloc_w(s.size1()); // for casadi_project
          sz_buf_k += s.nnz();
        }
      }

      // Only need the largest of these work vectors
      sz_buf = max(sz_buf, sz_buf_k);
    }

    // Memory for the work vectors
    alloc_w(sz_buf, true);
  }

  void Switch::eval(void* mem, const double** arg, double** res, int* iw, double* w) const {
    // Shorthands
    int n_in=this->n_in()-1, n_out=this->n_out();

    // Get the function to be evaluated
    int k = static_cast<int>(*arg[0]);
    const Function& fk = k<f_.size() ? f_[k] : f_def_;

    // Input and output buffers
    const double** arg1 = arg + 1 + n_in;
    copy_n(arg+1, n_in, arg1);
    double** res1 = res + n_out;
    copy_n(res, n_out, res1);

    // Project arguments with different sparsity
    for (int i=0; i<n_in; ++i) {
      if (arg1[i]) {
        const Sparsity& f_sp = fk.sparsity_in(i);
        const Sparsity& sp = sparsity_in(i+1);
        if (f_sp!=sp) {
          double *t = w; w += f_sp.nnz(); // t is non-const
          casadi_project(arg1[i], sp, t, f_sp, w);
          arg1[i] = t;
        }
      }
    }

    // Temporary memory for results with different sparsity
    for (int i=0; i<n_out; ++i) {
      if (res1[i]) {
        const Sparsity& f_sp = fk.sparsity_out(i);
        const Sparsity& sp = sparsity_out(i);
        if (f_sp!=sp) { res1[i] = w; w += f_sp.nnz();}
      }
    }

    // Evaluate the corresponding function
    fk(arg1, res1, iw, w, 0);

    // Project results with different sparsity
    for (int i=0; i<n_out; ++i) {
      if (res1[i]) {
        const Sparsity& f_sp = fk.sparsity_out(i);
        const Sparsity& sp = sparsity_out(i);
        if (f_sp!=sp) casadi_project(res1[i], f_sp, res[i], sp, w);
      }
    }
  }

  Function Switch
  ::get_forward(const std::string& name, int nfwd,
                const std::vector<std::string>& i_names,
                const std::vector<std::string>& o_names,
                const Dict& opts) const {
    // Derivative of each case
    vector<Function> der(f_.size());
    for (int k=0; k<f_.size(); ++k) {
      if (!f_[k].is_null()) der[k] = f_[k].forward(nfwd);
    }

    // Default case
    Function der_def;
    if (!f_def_.is_null()) der_def = f_def_.forward(nfwd);

    // New Switch for derivatives
    Function sw = Function::conditional("switch_" + name, der, der_def);

    // Get expressions for the derivative switch
    vector<MX> arg = sw.mx_in();
    vector<MX> res = sw(arg);

    // Ignore seed for ind
    arg.insert(arg.begin() + n_in() + n_out(), MX(1, nfwd));

    // Create wrapper
    return Function(name, arg, res, i_names, o_names, opts);
  }

  Function Switch
  ::get_reverse(const std::string& name, int nadj,
                const std::vector<std::string>& i_names,
                const std::vector<std::string>& o_names,
                const Dict& opts) const {
    // Derivative of each case
    vector<Function> der(f_.size());
    for (int k=0; k<f_.size(); ++k) {
      if (!f_[k].is_null()) der[k] = f_[k].reverse(nadj);
    }

    // Default case
    Function der_def;
    if (!f_def_.is_null()) der_def = f_def_.reverse(nadj);

    // New Switch for derivatives
    Function sw = Function::conditional("switch_" + name, der, der_def);

    // Get expressions for the derivative switch
    vector<MX> arg = sw.mx_in();
    vector<MX> res = sw(arg);

    // No derivatives with respect to index
    res.insert(res.begin(), MX(1, nadj));

    // Create wrapper
    return Function(name, arg, res, i_names, o_names, opts);
  }

  void Switch::print(ostream &stream) const {
    if (f_.size()==1) {
      // Print as if-then-else
      stream << "Switch(" << f_def_.name() << ", " << f_[0].name() << ")";
    } else {
      // Print generic
      stream << "Switch([";
      for (int k=0; k<f_.size(); ++k) {
        if (k!=0) stream << ", ";
        stream << f_[k].name();
      }
      stream << "], " << f_def_.name() << ")";
    }
  }

  void Switch::generateDeclarations(CodeGenerator& g) const {
    for (int k=0; k<=f_.size(); ++k) {
      const Function& fk = k<f_.size() ? f_[k] : f_def_;
      fk->addDependency(g);
    }
  }

  void Switch::eval_sx(const SXElem** arg, SXElem** res, int* iw, SXElem* w, int mem) const {

    // Shorthands
    int n_in=this->n_in()-1, n_out=this->n_out();

    // Input and output buffers
    const SXElem** arg1 = arg + 1 + n_in;
    SXElem** res1 = res + n_out;

    // Extra memory needed for chaining if_else calls
    std::vector<SXElem> w_extra(nnz_out());
    std::vector<SXElem*> res_tempv(n_out);
    SXElem** res_temp = get_ptr(res_tempv);

    for (int k=0; k<f_.size()+1; ++k) {

      // Local work vector
      SXElem* wl = w;

      // Local work vector
      SXElem* wll = get_ptr(w_extra);

      if (k==0) {
        // For the default case, redirect the temporary results to res
        copy_n(res, n_out, res_temp);
      } else {
        // For the other cases, store the temporary results
        for (int i=0; i<n_out; ++i) {
          res_temp[i] = wll;
          wll += nnz_out(i);
        }
      }

      copy_n(arg+1, n_in, arg1);
      copy_n(res_temp, n_out, res1);

      const Function& fk = k==0 ? f_def_ : f_[k-1];

      // Project arguments with different sparsity
      for (int i=0; i<n_in; ++i) {
        if (arg1[i]) {
          const Sparsity& f_sp = fk.sparsity_in(i);
          const Sparsity& sp = sparsity_in(i+1);
          if (f_sp!=sp) {
            SXElem *t = wl; wl += f_sp.nnz(); // t is non-const
            casadi_project(arg1[i], sp, t, f_sp, wl);
            arg1[i] = t;
          }
        }
      }

      // Temporary memory for results with different sparsity
      for (int i=0; i<n_out; ++i) {
        if (res1[i]) {
          const Sparsity& f_sp = fk.sparsity_out(i);
          const Sparsity& sp = sparsity_out(i);
          if (f_sp!=sp) { res1[i] = wl; wl += f_sp.nnz();}
        }
      }

      // Evaluate the corresponding function
      fk(arg1, res1, iw, wl, 0);

      // Project results with different sparsity
      for (int i=0; i<n_out; ++i) {
        if (res1[i]) {
          const Sparsity& f_sp = fk.sparsity_out(i);
          const Sparsity& sp = sparsity_out(i);
          if (f_sp!=sp) casadi_project(res1[i], f_sp, res_temp[i], sp, wl);
        }
      }

      if (k>0) { // output the temporary results via an if_else
        SXElem cond = k-1==arg[0][0];
        for (int i=0; i<n_out; ++i) {
          if (res[i]) {
            for (int j=0; j<nnz_out(i); ++j) {
              res[i][j] = if_else(cond, res_temp[i][j], res[i][j], true);
            }
          }
        }
      }

    }

  }

  void Switch::generateBody(CodeGenerator& g) const {
    // Shorthands
    int n_in=this->n_in()-1, n_out=this->n_out();

    // Input and output buffers
    g.body << "  int i;" << endl
           << "  double* t;" << endl
           << "  const double** arg1 = arg + " << (1 + n_in) << ";" << endl
           << "  for (i=0; i<" << n_in << "; ++i) arg1[i] = arg[1+i];" << endl
           << "  double** res1 = res + " << n_out << ";" << endl
           << "  for (i=0; i<" << n_out << "; ++i) res1[i] = res[i];" << endl;

    // Codegen condition
    bool if_else = f_.size()==1;
    g.body << "  " << (if_else ? "if" : "switch")  << " (to_int(arg[0][0])) {" << endl;

    // Loop over cases/functions
    for (int k=0; k<=f_.size(); ++k) {

      // For if,  reverse order
      int k1 = if_else ? 1-k : k;

      if (!if_else) {
        // Codegen cases
        if (k1<f_.size()) {
          g.body << "  case " << k1 << ":" << endl;
        } else {
          g.body << "  default:" << endl;
        }
      } else if (k1==0) {
        // Else
        g.body << "  } else {" << endl;
      }

      // Get the function:
      const Function& fk = k1<f_.size() ? f_[k1] : f_def_;
      if (fk.is_null()) {
        g.body << "    return 1;" << endl;
      } else if (g.simplifiedCall(fk)) {
        casadi_error("Not implemented.");
      } else {
        // Project arguments with different sparsity
        for (int i=0; i<n_in; ++i) {
          const Sparsity& f_sp = fk.sparsity_in(i);
          const Sparsity& sp = sparsity_in(i+1);
          if (f_sp!=sp) {
            g.body << "    if (arg1[" << i << "]) {" << endl
                   << "      t = w, w += " << f_sp.nnz() << ";" << endl
                   << "      " << g.project("arg1[" + to_string(i) + "]", sp,
                                            "t", f_sp, "w") << endl
                   << "      arg1[" << i << "] = t;" << endl
                   << "    }" << endl;
            }
          }

          // Temporary memory for results with different sparsity
          for (int i=0; i<n_out; ++i) {
            const Sparsity& f_sp = fk.sparsity_out(i);
            const Sparsity& sp = sparsity_out(i);
            if (f_sp!=sp) {
              g.body << "    if (res1[" << i << "]) {"
                     << "res1[" << i << "] = w; w += " << f_sp.nnz() << ";}" << endl;
            }
          }

          // Function call
          g.body << "    if (" << g(fk, "arg1", "res1", "iw", "w") << ") return 1;" << endl;

          // Project results with different sparsity
          for (int i=0; i<n_out; ++i) {
            const Sparsity& f_sp = fk.sparsity_out(i);
            const Sparsity& sp = sparsity_out(i);
            if (f_sp!=sp) {
              g.body << "    if (res[" << i << "]) "
                     << g.project("res1[" + to_string(i) + "]", f_sp,
                                  "res[" + to_string(i) + "]", sp, "w") << endl;
            }
          }

          // Break (if switch)
          if (!if_else) g.body << "    break;" << endl;
       }
    }

    // End switch/else
    g.body << "  }" << endl;
  }

} // namespace casadi
