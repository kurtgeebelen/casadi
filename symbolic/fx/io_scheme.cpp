/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010 by Joel Andersson, Moritz Diehl, K.U.Leuven. All rights reserved.
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

#include "io_scheme.hpp"
#include "io_scheme_internal.hpp"

namespace CasADi{

  IOScheme::IOScheme(){
  }
  
  IOScheme::IOScheme(InputOutputScheme scheme) {
    assignNode(new IOSchemeBuiltinInternal(scheme));
  }
  
  IOScheme::IOScheme(const std::vector<std::string> &entries) {
    assignNode(new IOSchemeCustomInternal(entries));
  }
 
  IOSchemeInternal* IOScheme::operator->(){
    return (IOSchemeInternal*)(SharedObject::operator->());
  }

  const IOSchemeInternal* IOScheme::operator->() const{
    return (const IOSchemeInternal*)(SharedObject::operator->());
  }
  
  bool IOScheme::checkNode() const{
    return dynamic_cast<const IOScheme*>(get())!=0;
  }

  std::string IOScheme::name() const {
    if (isNull()) return "Unknown";
    return (*this)->name();
  }
    
  std::string IOScheme::entryNames() const {
    if (isNull()) return "Not available";
    return (*this)->entryNames();
  }
    
  int IOScheme::index(const std::string &name) const {
    if (isNull()) casadi_error("Unknown scheme");
    return (*this)->index(name);
  }
    
  int IOScheme::size() const {
    if (isNull()) casadi_error("Unknown scheme has no known size.");
    return (*this)->size();
  }
  
  std::string IOScheme::entry(int i) const {
    if (isNull()) return "none";
    return (*this)->entry(i);
  }

  std::string IOScheme::entryEnum(int i) const {
    if (isNull()) return "none";
    return (*this)->entryEnum(i);
  }
  
  std::string IOScheme::describeInput(int i) const {
    if (isNull()) {
      std::stringstream ss;
      ss << "Input argument #" << i;
      return ss.str(); 
    }
    return (*this)->describeInput(i);
  }

  std::string IOScheme::describeOutput(int i) const {
    if (isNull()) {
      std::stringstream ss;
      ss << "Output argument #" << i;
      return ss.str(); 
    }
    return (*this)->describeOutput(i);
  }
  
  void IOScheme::print(std::ostream &stream) const {
    if (isNull()) {
      stream << "UnknownScheme";
      return;
    }
    return (*this)->print(stream);
  }

  void IOScheme::repr(std::ostream &stream) const {
    if (isNull()) {
      stream << "UknownScheme";
      return;
    }
    return (*this)->repr(stream);
  }

} // namespace CasADi