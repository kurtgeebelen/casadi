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


#ifndef CASADI_KERNEL_SUM_2D_INTERNAL_HPP
#define CASADI_KERNEL_SUM_2D_INTERNAL_HPP

#include "kernel_sum_2d.hpp"
#include "function_internal.hpp"

#ifdef WITH_OPENCL

#include "CL/cl.h"

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#endif // WITH_OPENCL

/// \cond INTERNAL

namespace casadi {


  class CASADI_EXPORT KernelSum2DBase : public FunctionInternal {
    friend class KernelSum2D;
  public:
    // Create function (use instead of constructor)
    static KernelSum2DBase* create(const Function& f,
           const std::pair<int, int> & size,
           double r,
           int n, const Dict& opts);

    /** \brief  Destructor */
    virtual ~KernelSum2DBase();

    /** \brief  Initialize */
    virtual void init();

    virtual void spFwd(const bvec_t** arg, bvec_t** res, int* iw, bvec_t* w);

    /** \brief  Is the class able to propagate seeds through the algorithm? */
    virtual bool spCanEvaluate(bool fwd) { return fwd; }
    
    ///@{
    /** \brief Generate a function that calculates \a nfwd forward derivatives */
    virtual Function getDerForward(const std::string& name, int nfwd, Dict& opts);
    virtual int numDerForward() const { return 64;}
    ///@}

    ///@{
    /** \brief Generate a function that calculates \a nadj adjoint derivatives */
    virtual Function getDerReverse(const std::string& name, int nadj, Dict& opts);
    virtual int numDerReverse() const { return 64;}
    ///@}
    
    
    /// Type of parallellization
    virtual std::string parallelization() const = 0;

  protected:
    // Constructor (protected, use create function above)
    KernelSum2DBase(const Function& f,
           const std::pair<int, int> & size,
           double r,
           int n);

    // Default case;
    Function f_;

    std::pair<int, int> size_;

    double r_;

    int n_;

    int nnz_out_;

    /// Nonzero step for outputs
    std::vector<int> step_out_;

  };

  /** KernelSum2D statement
      \author Joris Gillis
      \date 2015
  */
  class CASADI_EXPORT KernelSum2DSerial : public KernelSum2DBase {
  public:

    /** \brief Constructor (generic kernel_sum_2d) */
    KernelSum2DSerial(const Function& f,
           const std::pair<int, int> & size,
           double r,
           int n) : KernelSum2DBase(f, size, r, n) {}

    /** \brief  clone function */
    virtual KernelSum2DSerial* clone() const { return new KernelSum2DSerial(*this);}

    /** \brief  Destructor */
    virtual ~KernelSum2DSerial();

    /** \brief  Initialize */
    virtual void init();

    /** \brief  Evaluate numerically, work vectors given */
    virtual void evalD(const double** arg, double** res, int* iw, double* w);

    /** \brief  Print description */
    virtual void print(std::ostream &stream) const;

    /** \brief Generate code for the declarations of the C function */
    virtual void generateDeclarations(CodeGenerator& g) const;

    /** \brief Generate code for the body of the C function */
    virtual void generateBody(CodeGenerator& g) const;
    
    /// Type of parallellization
    virtual std::string parallelization() const { return "serial"; }

  };

#ifdef WITH_OPENCL
  /** KernelSum2D statement
      \author Joris Gillis
      \date 2015
  */
  class CASADI_EXPORT KernelSum2DOcl : public KernelSum2DBase {
  public:

    /** \brief Constructor (generic kernel_sum_2d) */
    KernelSum2DOcl(const Function& f,
           const std::pair<int, int> & size,
           double r,
           int n);

    /** \brief  clone function */
    virtual KernelSum2DOcl* clone() const { return new KernelSum2DOcl(*this);}

    /** \brief  Destructor */
    virtual ~KernelSum2DOcl();

    /** \brief  Initialize */
    virtual void init();

    /** \brief  Evaluate numerically, work vectors given */
    virtual void evalD(const double** arg, double** res, int* iw, double* w);
    
    /// Type of parallellization
    virtual std::string parallelization() const { return "opencl"; }
    
    /// Obtain code for the kernel
    std::string kernelCode() const;

    std::vector<cl_device_id> devices_;

    cl_context context_;
/**
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> *kernel_;
    cl::CommandQueue queue_;

    cl::Buffer d_im_;
    cl::Buffer d_args_;
    cl::Buffer d_sum_;
*/
    std::vector< float > h_sum_;
    std::vector< float > h_args_;
    std::vector< float > h_im_;

    int nnz_in_;
    int nnz_fixed_;
    
    int s_;
    int ss_;

    double sfrac_;


    std::vector<int> opencl_select_;

    /** \brief Generate code for the declarations of the C function */
    virtual void generateDeclarations(CodeGenerator& g) const;

    /** \brief Generate code for the body of the C function */
    virtual void generateBody(CodeGenerator& g) const;

  };

#endif // WITH_OPENCL

} // namespace casadi
/// \endcond

#endif // CASADI_KERNEL_SUM_2D_INTERNAL_HPP
