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


#include "casadi_find.hpp"

using namespace std;

namespace casadi {

  Find::Find(const MX& x) {
    casadi_assert(x.is_column());
    setDependencies(x);
    setSparsity(Sparsity::scalar());
  }

  std::string Find::print(const std::vector<std::string>& arg) const {
    return "find(" + arg.at(0) + ")";
  }

  void Find::eval(const double** arg, double** res, int* iw, double* w, int mem) const {
    const double* x = arg[0];
    int nnz = dep(0).nnz();
    int k=0;
    while (k<nnz && *x++ == 0) k++;
    res[0][0] = k<nnz ? dep(0).row(k) : dep(0).size1();
  }

  void Find::eval_mx(const std::vector<MX>& arg, std::vector<MX>& res) const {
    res[0] = find(arg[0]);
  }

  void Find::eval_forward(const std::vector<std::vector<MX> >& fseed,
                     std::vector<std::vector<MX> >& fsens) const {
    for (int d=0; d<fsens.size(); ++d) {
      fsens[d][0] = 0;
    }
  }

  void Find::eval_reverse(const std::vector<std::vector<MX> >& aseed,
                     std::vector<std::vector<MX> >& asens) const {
  }

  void Find::sp_fwd(const bvec_t** arg, bvec_t** res, int* iw, bvec_t* w, int mem) const {
    res[0][0] = 0; // pw constant
  }

  void Find::sp_rev(bvec_t** arg, bvec_t** res, int* iw, bvec_t* w, int mem) const {
    res[0][0] = 0; // pw constant
  }

  void Find::generate(CodeGenerator& g, const std::string& mem,
                      const std::vector<int>& arg, const std::vector<int>& res) const {
    int nnz = dep(0).nnz();
    g.body << "  for (i=0, cr=" << g.work(arg[0], nnz) << "; i<" << nnz
           << " && *cr++==0; ++i) {}" << endl
           << "  " << g.workel(res[0]) << " = ";
    if (dep(0).is_dense()) {
      g.body << "i" << ";" << endl;
    } else {
      // The row is in position 1+1+2+i (colind has length 2)
      g.body << "i<" << nnz << " ? " << g.sparsity(dep(0).sparsity()) << "[4+i] : "
             << dep(0).size1() << endl;
    }
  }

} // namespace casadi
