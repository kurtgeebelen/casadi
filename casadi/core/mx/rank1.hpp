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


#ifndef CASADI_RANK1_HPP
#define CASADI_RANK1_HPP

#include "mx_node.hpp"
#include <map>
#include <stack>

/// \cond INTERNAL

namespace casadi {
  /** \brief Calculate rank1 update
      \author Joel Andersson
      \date 2015
  */
  class CASADI_EXPORT Rank1 : public MXNode {
  public:

    /// Constructor
    Rank1(const MX& A, const MX& alpha, const MX& x, const MX& y);

    /// Destructor
    virtual ~Rank1() {}

    /// Evaluate the function (template)
    template<typename T>
    void evalGen(const T** arg, T** res, int* iw, T* w, int mem) const;

    /// Evaluate the function numerically
    virtual void eval(const double** arg, double** res, int* iw, double* w, int mem) const;

    /// Evaluate the function symbolically (SX)
    virtual void eval_sx(const SXElem** arg, SXElem** res, int* iw, SXElem* w, int mem) const;

    /** \brief  Propagate sparsity forward */
    virtual void sp_fwd(const bvec_t** arg, bvec_t** res, int* iw, bvec_t* w, int mem) const;

    /** \brief  Propagate sparsity backwards */
    virtual void sp_rev(bvec_t** arg, bvec_t** res, int* iw, bvec_t* w, int mem) const;

    /** \brief  Evaluate symbolically (MX) */
    virtual void eval_mx(const std::vector<MX>& arg, std::vector<MX>& res) const;

    /** \brief Calculate forward mode directional derivatives */
    virtual void eval_forward(const std::vector<std::vector<MX> >& fseed,
                         std::vector<std::vector<MX> >& fsens) const;

    /** \brief Calculate reverse mode directional derivatives */
    virtual void eval_reverse(const std::vector<std::vector<MX> >& aseed,
                         std::vector<std::vector<MX> >& asens) const;

    /** \brief Generate code for the operation */
    void generate(CodeGenerator& g, const std::string& mem,
                  const std::vector<int>& arg, const std::vector<int>& res) const;

    /// Can the operation be performed inplace (i.e. overwrite the result)
    virtual int numInplace() const { return 1;}

    /** \brief  Print expression */
    virtual std::string print(const std::vector<std::string>& arg) const;

    /** \brief Get the operation */
    virtual int op() const { return OP_RANK1;}
  };


} // namespace casadi
/// \endcond

#endif // CASADI_RANK1_HPP
