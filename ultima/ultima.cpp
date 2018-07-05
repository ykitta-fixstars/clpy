#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include <memory>
#include <utility>

namespace ultima{

namespace detail{

template<std::size_t N>struct priority : priority<N-1>{};
template<>struct priority<0>{};

struct hasPrintTemplateArgumentList{
  template<typename T, typename... Args>
  static auto check(priority<1>, Args&&...) -> decltype(T::PrintTemplateArgumentList(std::declval<Args>()...), std::true_type{});
  template<typename T, typename... Args>
  static std::false_type check(priority<0>, Args&&...);
};

template<typename T, typename... Args>
static inline auto printTemplateArgumentList(Args&&... args)->decltype(T::PrintTemplateArgumentList(std::forward<Args>(args)...)){
  return T::PrintTemplateArgumentList(std::forward<Args>(args)...);
}

}

}

namespace clang{

template<typename... Args, typename std::enable_if<decltype(ultima::detail::hasPrintTemplateArgumentList::check<clang::TemplateSpecializationType>(ultima::detail::priority<1>{}, std::declval<Args>()...))::value, std::nullptr_t>::type = nullptr>
static inline auto printTemplateArgumentList(Args&&... args)->decltype(ultima::detail::printTemplateArgumentList<clang::TemplateSpecializationType>(std::forward<Args>(args)...)){
  return ultima::detail::printTemplateArgumentList<clang::TemplateSpecializationType>(std::forward<Args>(args)...);
}

}

namespace ultima{

struct ostreams{
  std::vector<llvm::raw_ostream*> oss;
  ostreams(llvm::raw_ostream& os):oss{&os}{}
  template<typename T>
    llvm::raw_ostream& operator<<(T&& rhs){return (*oss.back()) << rhs;}
  operator llvm::raw_ostream&(){return *oss.back();}
  void push(llvm::raw_ostream& os){oss.emplace_back(&os);}
  void pop(){oss.pop_back();}
  struct auto_popper{
    ostreams* oss;
    auto_popper(ostreams& oss, llvm::raw_ostream& os):oss{&oss}{oss.push(os);}
    auto_popper(auto_popper&& other):oss{other.oss}{other.oss = nullptr;}
    ~auto_popper(){if(oss)oss->pop();}
  };
  auto_popper scoped_push(llvm::raw_ostream& os){return {*this, os};}
};

class decl_visitor;

template<typename T>
static inline void printGroup(clang::DeclVisitor<T>& t, clang::Decl** b, std::size_t s){
  static_cast<T&>(t).printGroup(b, s);
}
template<typename T, typename U>
static inline void prettyPrintAttributes(clang::DeclVisitor<T>& t, U* u){
  static_cast<T&>(t).prettyPrintAttributes(u);
}


class stmt_visitor : public clang::StmtVisitor<stmt_visitor> {
  ostreams& os;
  unsigned& IndentLevel;
  clang::PrintingPolicy& Policy;
  clang::DeclVisitor<decl_visitor>& dv;

public:
  stmt_visitor(ostreams& os,
              clang::PrintingPolicy &Policy,
              unsigned& Indentation, clang::DeclVisitor<decl_visitor>& dv)
    : os(os), IndentLevel(Indentation), Policy(Policy), dv{dv} {}

  void PrintStmt(clang::Stmt *S) {
    PrintStmt(S, Policy.Indentation);
  }

  void PrintStmt(clang::Stmt *S, int SubIndent) {
    IndentLevel += SubIndent;
    if (S && clang::isa<clang::Expr>(S)) {
      // If this is an expr used in a stmt context, indent and newline it.
      Indent();
      Visit(S);
      os << ";\n";
    } else if (S) {
      Visit(S);
    } else {
      Indent() << "/*<<<NULL STATEMENT>>>*/\n";
    }
    IndentLevel -= SubIndent;
  }

  void PrintExpr(clang::Expr *E) {
    if (E)
      Visit(E);
    else
      os << "/*<null expr>*/";
  }

  llvm::raw_ostream &Indent(int Delta = 0) {
    for (int i = 0, e = IndentLevel+Delta; i < e; ++i)
      os << "  ";
    return os;
  }

  void VisitStmt(clang::Stmt*) LLVM_ATTRIBUTE_UNUSED {
    Indent() << "<<unknown stmt type>>\n";
  }
  void VisitExpr(clang::Expr*) LLVM_ATTRIBUTE_UNUSED {
    os << "<<unknown expr type>>";
  }

  /// PrintRawCompoundStmt - Print a compound stmt without indenting the {, and
  /// with no newline after the }.
  void PrintRawCompoundStmt(clang::CompoundStmt *Node) {
    os << "{\n";
    for (auto *I : Node->body())
      PrintStmt(I);

    Indent() << '}';
  }

  void PrintRawDecl(clang::Decl *D) {
    dv.Visit(D);
  }

  void PrintRawDeclStmt(const clang::DeclStmt *S) {
    llvm::SmallVector<clang::Decl*, 2> Decls(S->decls());
    printGroup(dv, Decls.data(), Decls.size());
  }

  void VisitNullStmt(clang::NullStmt*) {
    Indent() << ";\n";
  }

  void VisitDeclStmt(clang::DeclStmt *Node) {
    Indent();
    PrintRawDeclStmt(Node);
    os << ";\n";
  }

  void VisitCompoundStmt(clang::CompoundStmt *Node) {
    Indent();
    PrintRawCompoundStmt(Node);
    os << "\n";
  }

  void VisitCaseStmt(clang::CaseStmt *Node) {
    Indent(-1) << "case ";
    PrintExpr(Node->getLHS());
    if (Node->getRHS()) {
      os << " ... ";
      PrintExpr(Node->getRHS());
    }
    os << ":\n";

    PrintStmt(Node->getSubStmt(), 0);
  }

  void VisitDefaultStmt(clang::DefaultStmt *Node) {
    Indent(-1) << "default:\n";
    PrintStmt(Node->getSubStmt(), 0);
  }

  void VisitLabelStmt(clang::LabelStmt *Node) {
    Indent(-1) << Node->getName() << ":\n";
    PrintStmt(Node->getSubStmt(), 0);
  }

  void VisitAttributedStmt(clang::AttributedStmt *Node) {
    prettyPrintAttributes(dv, Node);
    PrintStmt(Node->getSubStmt(), 0);
  }

  void PrintRawIfStmt(clang::IfStmt *If) {
    os << "if (";
    if (const auto *DS = If->getConditionVariableDeclStmt())
      PrintRawDeclStmt(DS);
    else
      PrintExpr(If->getCond());
    os << ')';

    if (auto *CS = clang::dyn_cast<clang::CompoundStmt>(If->getThen())) {
      os << ' ';
      PrintRawCompoundStmt(CS);
      os << (If->getElse() ? ' ' : '\n');
    } else {
      os << '\n';
      PrintStmt(If->getThen());
      if (If->getElse()) Indent();
    }

    if (auto *Else = If->getElse()) {
      os << "else";

      if (auto *CS = clang::dyn_cast<clang::CompoundStmt>(Else)) {
        os << ' ';
        PrintRawCompoundStmt(CS);
        os << '\n';
      } else if (auto *ElseIf = clang::dyn_cast<clang::IfStmt>(Else)) {
        os << ' ';
        PrintRawIfStmt(ElseIf);
      } else {
        os << '\n';
        PrintStmt(If->getElse());
      }
    }
  }

  void VisitIfStmt(clang::IfStmt *If) {
    Indent();
    PrintRawIfStmt(If);
  }

  void VisitSwitchStmt(clang::SwitchStmt *Node) {
    Indent() << "switch (";
    if (const auto *DS = Node->getConditionVariableDeclStmt())
      PrintRawDeclStmt(DS);
    else
      PrintExpr(Node->getCond());
    os << ')';

    // Pretty print compoundstmt bodies (very common).
    if (auto *CS = clang::dyn_cast<clang::CompoundStmt>(Node->getBody())) {
      os << ' ';
      PrintRawCompoundStmt(CS);
      os << '\n';
    } else {
      os << '\n';
      PrintStmt(Node->getBody());
    }
  }

  void VisitWhileStmt(clang::WhileStmt *Node) {
    Indent() << "while (";
    if (const auto *DS = Node->getConditionVariableDeclStmt())
      PrintRawDeclStmt(DS);
    else
      PrintExpr(Node->getCond());
    os << ")\n";
    PrintStmt(Node->getBody());
  }

  void VisitDoStmt(clang::DoStmt *Node) {
    Indent() << "do ";
    if (auto *CS = clang::dyn_cast<clang::CompoundStmt>(Node->getBody())) {
      PrintRawCompoundStmt(CS);
      os << ' ';
    } else {
      os << "\n";
      PrintStmt(Node->getBody());
      Indent();
    }

    os << "while (";
    PrintExpr(Node->getCond());
    os << ");\n";
  }

  void VisitForStmt(clang::ForStmt *Node) {
    Indent() << "for (";
    if (Node->getInit()) {
      if (auto *DS = clang::dyn_cast<clang::DeclStmt>(Node->getInit()))
        PrintRawDeclStmt(DS);
      else
        PrintExpr(clang::cast<clang::Expr>(Node->getInit()));
    }
    os << ';';
    if (Node->getCond()) {
      os << ' ';
      PrintExpr(Node->getCond());
    }
    os << ';';
    if (Node->getInc()) {
      os << ' ';
      PrintExpr(Node->getInc());
    }
    os << ") ";

    if (auto *CS = clang::dyn_cast<clang::CompoundStmt>(Node->getBody())) {
      PrintRawCompoundStmt(CS);
      os << '\n';
    } else {
      os << '\n';
      PrintStmt(Node->getBody());
    }
  }

  void VisitCXXForRangeStmt(clang::CXXForRangeStmt *Node) {
    Indent() << "for (";
    auto backup = Policy;
    Policy.SuppressInitializers = true;
    dv.Visit(Node->getLoopVariable());
    Policy = backup;
    os << " : ";
    PrintExpr(Node->getRangeInit());
    os << ") {\n";
    PrintStmt(Node->getBody());
    Indent() << '}';
    if (Policy.IncludeNewlines) os << '\n';
  }

  void VisitMSDependentExistsStmt(clang::MSDependentExistsStmt *Node) {
    Indent();
    if (Node->isIfExists())
      os << "__if_exists (";
    else
      os << "__if_not_exists (";
    
    if (auto *Qualifier
          = Node->getQualifierLoc().getNestedNameSpecifier())
      Qualifier->print(os, Policy);
    
    os << Node->getNameInfo() << ") ";
    
    PrintRawCompoundStmt(Node->getSubStmt());
  }

  void VisitGotoStmt(clang::GotoStmt *Node) {
    Indent() << "goto " << Node->getLabel()->getName() << ';';
    if (Policy.IncludeNewlines) os << '\n';
  }

  void VisitIndirectGotoStmt(clang::IndirectGotoStmt *Node) {
    Indent() << "goto *";
    PrintExpr(Node->getTarget());
    os << ';';
    if (Policy.IncludeNewlines) os << '\n';
  }

  void VisitContinueStmt(clang::ContinueStmt*) {
    Indent() << "continue;";
    if (Policy.IncludeNewlines) os << '\n';
  }

  void VisitBreakStmt(clang::BreakStmt*) {
    Indent() << "break;";
    if (Policy.IncludeNewlines) os << '\n';
  }


  void VisitReturnStmt(clang::ReturnStmt *Node) {
    Indent() << "return";
    if (Node->getRetValue()) {
      os << ' ';
      PrintExpr(Node->getRetValue());
    }
    os << ';';
    if (Policy.IncludeNewlines) os << '\n';
  }


  void VisitGCCAsmStmt(clang::GCCAsmStmt *Node) {
    Indent() << "asm ";

    if (Node->isVolatile())
      os << "volatile ";

    os << '(';
    VisitStringLiteral(Node->getAsmString());

    // Outputs
    if (Node->getNumOutputs() != 0 || Node->getNumInputs() != 0 ||
        Node->getNumClobbers() != 0)
      os << " : ";

    for (unsigned i = 0, e = Node->getNumOutputs(); i != e; ++i) {
      if (i != 0)
        os << ", ";

      if (!Node->getOutputName(i).empty()) {
        os << '[';
        os << Node->getOutputName(i);
        os << "] ";
      }

      VisitStringLiteral(Node->getOutputConstraintLiteral(i));
      os << " (";
      Visit(Node->getOutputExpr(i));
      os << ')';
    }

    // Inputs
    if (Node->getNumInputs() != 0 || Node->getNumClobbers() != 0)
      os << " : ";

    for (unsigned i = 0, e = Node->getNumInputs(); i != e; ++i) {
      if (i != 0)
        os << ", ";

      if (!Node->getInputName(i).empty()) {
        os << '[';
        os << Node->getInputName(i);
        os << "] ";
      }

      VisitStringLiteral(Node->getInputConstraintLiteral(i));
      os << " (";
      Visit(Node->getInputExpr(i));
      os << ')';
    }

    // Clobbers
    if (Node->getNumClobbers() != 0)
      os << " : ";

    for (unsigned i = 0, e = Node->getNumClobbers(); i != e; ++i) {
      if (i != 0)
        os << ", ";

      VisitStringLiteral(Node->getClobberStringLiteral(i));
    }

    os << ");";
    if (Policy.IncludeNewlines) os << '\n';
  }

  void VisitMSAsmStmt(clang::MSAsmStmt *Node) {
    // FIXME: Implement MS style inline asm statement printer.
    Indent() << "__asm ";
    if (Node->hasBraces())
      os << "{\n";
    os << Node->getAsmString() << '\n';
    if (Node->hasBraces())
      Indent() << "}\n";
  }

  void VisitCapturedStmt(clang::CapturedStmt *Node) {
    PrintStmt(Node->getCapturedDecl()->getBody());
  }

  void PrintRawCXXCatchStmt(clang::CXXCatchStmt *Node) {
    os << "catch (";
    if (clang::Decl *ExDecl = Node->getExceptionDecl())
      PrintRawDecl(ExDecl);
    else
      os << "...";
    os << ") ";
    PrintRawCompoundStmt(clang::cast<clang::CompoundStmt>(Node->getHandlerBlock()));
  }

  void VisitCXXCatchStmt(clang::CXXCatchStmt *Node) {
    Indent();
    PrintRawCXXCatchStmt(Node);
    os << '\n';
  }

  void VisitCXXTryStmt(clang::CXXTryStmt *Node) {
    Indent() << "try ";
    PrintRawCompoundStmt(Node->getTryBlock());
    for (unsigned i = 0, e = Node->getNumHandlers(); i < e; ++i) {
      os << ' ';
      PrintRawCXXCatchStmt(Node->getHandler(i));
    }
    os << '\n';
  }

  void VisitDeclRefExpr(clang::DeclRefExpr *Node) {
    if (auto *Qualifier = Node->getQualifier())
      Qualifier->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }

  void VisitDependentScopeDeclRefExpr(clang::DependentScopeDeclRefExpr *Node) {
    if (auto *Qualifier = Node->getQualifier())
      Qualifier->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }

  void VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr *Node) {
    if (Node->getQualifier())
      Node->getQualifier()->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }

  void VisitPredefinedExpr(clang::PredefinedExpr *Node) {
    os << clang::PredefinedExpr::getIdentTypeName(Node->getIdentType());
  }

  void VisitCharacterLiteral(clang::CharacterLiteral *Node) {
    Node->printPretty(os, nullptr, Policy);
  }

  void VisitIntegerLiteral(clang::IntegerLiteral *Node) {
    Node->printPretty(os, nullptr, Policy);
  }

  static void PrintFloatingLiteral(llvm::raw_ostream &os, clang::FloatingLiteral *Node,
                                   bool PrintSuffix) {
    llvm::SmallString<16> Str;
    Node->getValue().toString(Str);
    os << Str;
    if (Str.find_first_not_of("-0123456789") == StringRef::npos)
      os << '.'; // Trailing dot in order to separate from ints.

    if (!PrintSuffix)
      return;

    // Emit suffixes.  Float literals are always a builtin float type.
    switch (Node->getType()->getAs<clang::BuiltinType>()->getKind()) {
    default: llvm_unreachable("Unexpected type for float literal!");
    case clang::BuiltinType::Half:       break; // FIXME: suffix?
    case clang::BuiltinType::Double:     break; // no suffix.
    case clang::BuiltinType::Float:      os << 'F'; break;
    case clang::BuiltinType::LongDouble: os << 'L'; break;
    case clang::BuiltinType::Float128:   os << 'Q'; break;
    }
  }

  void VisitFloatingLiteral(clang::FloatingLiteral *Node) {
    PrintFloatingLiteral(os, Node, /*PrintSuffix=*/true);
  }

  void VisitImaginaryLiteral(clang::ImaginaryLiteral *Node) {
    PrintExpr(Node->getSubExpr());
    os << 'i';
  }

  void VisitStringLiteral(clang::StringLiteral *Str) {
    Str->outputString(os);
  }
  void VisitParenExpr(clang::ParenExpr *Node) {
    os << '(';
    PrintExpr(Node->getSubExpr());
    os << ')';
  }
  void VisitUnaryOperator(clang::UnaryOperator *Node) {
    if (!Node->isPostfix()) {
      os << clang::UnaryOperator::getOpcodeStr(Node->getOpcode());

      // Print a space if this is an "identifier operator" like __real, or if
      // it might be concatenated incorrectly like '+'.
      switch (Node->getOpcode()) {
      default: break;
      case clang::UO_Real:
      case clang::UO_Imag:
      case clang::UO_Extension:
        os << ' ';
        break;
      case clang::UO_Plus:
      case clang::UO_Minus:
        if (clang::isa<clang::UnaryOperator>(Node->getSubExpr()))
          os << ' ';
        break;
      }
    }
    PrintExpr(Node->getSubExpr());

    if (Node->isPostfix())
      os << clang::UnaryOperator::getOpcodeStr(Node->getOpcode());
  }

  void VisitOffsetOfExpr(clang::OffsetOfExpr *Node) {
    os << "__builtin_offsetof(";
    Node->getTypeSourceInfo()->getType().print(os, Policy);
    os << ", ";
    bool PrintedSomething = false;
    for (unsigned i = 0, n = Node->getNumComponents(); i < n; ++i) {
      auto ON = Node->getComponent(i);
      if (ON.getKind() == clang::OffsetOfNode::Array) {
        // Array node
        os << '[';
        PrintExpr(Node->getIndexExpr(ON.getArrayExprIndex()));
        os << ']';
        PrintedSomething = true;
        continue;
      }

      // Skip implicit base indirections.
      if (ON.getKind() == clang::OffsetOfNode::Base)
        continue;

      // Field or identifier node.
      auto *Id = ON.getFieldName();
      if (!Id)
        continue;
      
      if (PrintedSomething)
        os << '.';
      else
        PrintedSomething = true;
      os << Id->getName();    
    }
    os << ')';
  }

  void VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *Node){
    switch(Node->getKind()) {
    case clang::UETT_SizeOf:
      os << "sizeof";
      break;
    case clang::UETT_AlignOf:
      if (Policy.Alignof)
        os << "alignof";
      else if (Policy.UnderscoreAlignof)
        os << "_Alignof";
      else
        os << "__alignof";
      break;
    case clang::UETT_VecStep:
      os << "vec_step";
      break;
    default: llvm_unreachable("OpenMP is not supported.");
    }
    if (Node->isArgumentType()) {
      os << '(';
      Node->getArgumentType().print(os, Policy);
      os << ')';
    } else {
      os << ' ';
      PrintExpr(Node->getArgumentExpr());
    }
  }

  void VisitGenericSelectionExpr(clang::GenericSelectionExpr *Node) {
    os << "_Generic(";
    PrintExpr(Node->getControllingExpr());
    for (unsigned i = 0; i != Node->getNumAssocs(); ++i) {
      os << ", ";
      auto T = Node->getAssocType(i);
      if (T.isNull())
        os << "default";
      else
        T.print(os, Policy);
      os << ": ";
      PrintExpr(Node->getAssocExpr(i));
    }
    os << ')';
  }

  void VisitArraySubscriptExpr(clang::ArraySubscriptExpr *Node) {
    PrintExpr(Node->getLHS());
    os << '[';
    PrintExpr(Node->getRHS());
    os << ']';
  }

  void PrintCallArgs(clang::CallExpr *Call) {
    for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
      if (clang::isa<clang::CXXDefaultArgExpr>(Call->getArg(i))) {
        // Don't print any defaulted arguments
        break;
      }

      if (i) os << ", ";
      PrintExpr(Call->getArg(i));
    }
  }

  void VisitCallExpr(clang::CallExpr *Call) {
    PrintExpr(Call->getCallee());
    os << '(';
    PrintCallArgs(Call);
    os << ')';
  }
  void VisitMemberExpr(clang::MemberExpr *Node) {
    // FIXME: Suppress printing implicit bases (like "this")
    PrintExpr(Node->getBase());

    auto *ParentMember = clang::dyn_cast<clang::MemberExpr>(Node->getBase());
    auto *ParentDecl   = ParentMember
      ? clang::dyn_cast<clang::FieldDecl>(ParentMember->getMemberDecl()) : nullptr;

    if (!ParentDecl || !ParentDecl->isAnonymousStructOrUnion()){
      if(Node->isArrow())
        os << "->";
      else
        os << '.';
    }

    if (clang::FieldDecl *FD = clang::dyn_cast<clang::FieldDecl>(Node->getMemberDecl()))
      if (FD->isAnonymousStructOrUnion())
        return;

    if (auto *Qualifier = Node->getQualifier())
      Qualifier->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getMemberNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }
  void VisitExtVectorElementExpr(clang::ExtVectorElementExpr *Node) {
    PrintExpr(Node->getBase());
    os << '.';
    os << Node->getAccessor().getName();
  }
  void VisitCStyleCastExpr(clang::CStyleCastExpr *Node) {
    os << '(';
    Node->getTypeAsWritten().print(os, Policy);
    os << ')';
    PrintExpr(Node->getSubExpr());
  }
  void VisitCompoundLiteralExpr(clang::CompoundLiteralExpr *Node) {
    os << '(';
    Node->getType().print(os, Policy);
    os << ')';
    PrintExpr(Node->getInitializer());
  }
  void VisitImplicitCastExpr(clang::ImplicitCastExpr *Node) {
    PrintExpr(Node->getSubExpr());
  }
  void VisitBinaryOperator(clang::BinaryOperator *Node) {
    PrintExpr(Node->getLHS());
    os << ' ' << clang::BinaryOperator::getOpcodeStr(Node->getOpcode()) << ' ';
    PrintExpr(Node->getRHS());
  }
  void VisitCompoundAssignOperator(clang::CompoundAssignOperator *Node) {
    PrintExpr(Node->getLHS());
    os << ' ' << clang::BinaryOperator::getOpcodeStr(Node->getOpcode()) << ' ';
    PrintExpr(Node->getRHS());
  }
  void VisitConditionalOperator(clang::ConditionalOperator *Node) {
    PrintExpr(Node->getCond());
    os << " ? ";
    PrintExpr(Node->getLHS());
    os << " : ";
    PrintExpr(Node->getRHS());
  }

  // GNU extensions.

  void VisitBinaryConditionalOperator(clang::BinaryConditionalOperator *Node) {
    PrintExpr(Node->getCommon());
    os << " ?: ";
    PrintExpr(Node->getFalseExpr());
  }
  void VisitAddrLabelExpr(clang::AddrLabelExpr *Node) {
    os << "&&" << Node->getLabel()->getName();
  }

  void VisitStmtExpr(clang::StmtExpr *E) {
    os << '(';
    PrintRawCompoundStmt(E->getSubStmt());
    os << ')';
  }

  void VisitChooseExpr(clang::ChooseExpr *Node) {
    os << "__builtin_choose_expr(";
    PrintExpr(Node->getCond());
    os << ", ";
    PrintExpr(Node->getLHS());
    os << ", ";
    PrintExpr(Node->getRHS());
    os << ')';
  }

  void VisitGNUNullExpr(clang::GNUNullExpr *) {
    os << "__null";
  }

  void VisitShuffleVectorExpr(clang::ShuffleVectorExpr *Node) {
    os << "__builtin_shufflevector(";
    for (unsigned i = 0, e = Node->getNumSubExprs(); i != e; ++i) {
      if (i) os << ", ";
      PrintExpr(Node->getExpr(i));
    }
    os << ')';
  }

  void VisitConvertVectorExpr(clang::ConvertVectorExpr *Node) {
    os << "__builtin_convertvector(";
    PrintExpr(Node->getSrcExpr());
    os << ", ";
    Node->getType().print(os, Policy);
    os << ')';
  }

  void VisitInitListExpr(clang::InitListExpr* Node) {
    if (Node->getSyntacticForm()) {
      Visit(Node->getSyntacticForm());
      return;
    }

    os << '{';
    for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
      if (i) os << ", ";
      if (Node->getInit(i))
        PrintExpr(Node->getInit(i));
      else
        os << "{}";
    }
    os << '}';
  }

  void VisitArrayInitLoopExpr(clang::ArrayInitLoopExpr *Node) {
    // There's no way to express this expression in any of our supported
    // languages, so just emit something terse and (hopefully) clear.
    os << '{';
    PrintExpr(Node->getSubExpr());
    os << '}';
  }

  void VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr*) {
    os << '*';
  }

  void VisitParenListExpr(clang::ParenListExpr* Node) {
    os << '(';
    for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
      if (i) os << ", ";
      PrintExpr(Node->getExpr(i));
    }
    os << ')';
  }

  void VisitDesignatedInitExpr(clang::DesignatedInitExpr *Node) {
    bool NeedsEquals = true;
    for (const auto &D : Node->designators()) {
      if (D.isFieldDesignator()) {
        if (D.getDotLoc().isInvalid()) {
          if (auto *II = D.getFieldName()) {
            os << II->getName() << ':';
            NeedsEquals = false;
          }
        } else {
          os << '.' << D.getFieldName()->getName();
        }
      } else {
        os << '[';
        if (D.isArrayDesignator()) {
          PrintExpr(Node->getArrayIndex(D));
        } else {
          PrintExpr(Node->getArrayRangeStart(D));
          os << " ... ";
          PrintExpr(Node->getArrayRangeEnd(D));
        }
        os << ']';
      }
    }

    if (NeedsEquals)
      os << " = ";
    else
      os << ' ';
    PrintExpr(Node->getInit());
  }

  void VisitDesignatedInitUpdateExpr(
      clang::DesignatedInitUpdateExpr *Node) {
    os << '{';
    os << "/*base*/";
    PrintExpr(Node->getBase());
    os << ", ";

    os << "/*updater*/";
    PrintExpr(Node->getUpdater());
    os << '}';
  }

  void VisitNoInitExpr(clang::NoInitExpr*) {}

  void VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *Node) {
    if (Node->getType()->getAsCXXRecordDecl()) {
      os << "/*implicit*/";
      Node->getType().print(os, Policy);
      os << "()";
    } else {
      os << "/*implicit*/(";
      Node->getType().print(os, Policy);
      os << ')';
      if (Node->getType()->isRecordType())
        os << "{}";
      else
        os << 0;
    }
  }

  void VisitVAArgExpr(clang::VAArgExpr *Node) {
    os << "__builtin_va_arg(";
    PrintExpr(Node->getSubExpr());
    os << ", ";
    Node->getType().print(os, Policy);
    os << ')';
  }

  void VisitPseudoObjectExpr(clang::PseudoObjectExpr *Node) {
    PrintExpr(Node->getSyntacticForm());
  }

  void VisitAtomicExpr(clang::AtomicExpr *Node) {
    const char *Name = nullptr;
    switch (Node->getOp()) {
      using namespace clang;
#define BUILTIN(ID, TYPE, ATTRS)
#define ATOMIC_BUILTIN(ID, TYPE, ATTRS) \
    case AtomicExpr::AO ## ID: \
      Name = #ID "("; \
      break;
#include "clang/Basic/Builtins.def"
#undef ATOMIC_BUILTIN
#undef BUILTIN
    }
    os << Name;

    // AtomicExpr stores its subexpressions in a permuted order.
    PrintExpr(Node->getPtr());
    if (Node->getOp() != clang::AtomicExpr::AO__c11_atomic_load &&
        Node->getOp() != clang::AtomicExpr::AO__atomic_load_n) {
      os << ", ";
      PrintExpr(Node->getVal1());
    }
    if (Node->getOp() == clang::AtomicExpr::AO__atomic_exchange ||
        Node->isCmpXChg()) {
      os << ", ";
      PrintExpr(Node->getVal2());
    }
    if (Node->getOp() == clang::AtomicExpr::AO__atomic_compare_exchange ||
        Node->getOp() == clang::AtomicExpr::AO__atomic_compare_exchange_n) {
      os << ", ";
      PrintExpr(Node->getWeak());
    }
    if (Node->getOp() != clang::AtomicExpr::AO__c11_atomic_init) {
      os << ", ";
      PrintExpr(Node->getOrder());
    }
    if (Node->isCmpXChg()) {
      os << ", ";
      PrintExpr(Node->getOrderFail());
    }
    os << ')';
  }

  // C++
  void VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *Node) {
    const char *OpStrings[clang::NUM_OVERLOADED_OPERATORS] = {
      "",
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
      Spelling,
#include "clang/Basic/OperatorKinds.def"
#undef OVERLOADED_OPERATOR
    };

    auto Kind = Node->getOperator();
    if (Kind == clang::OO_PlusPlus || Kind == clang::OO_MinusMinus) {
      if (Node->getNumArgs() == 1) {
        os << OpStrings[Kind] << ' ';
        PrintExpr(Node->getArg(0));
      } else {
        PrintExpr(Node->getArg(0));
        os << ' ' << OpStrings[Kind];
      }
    } else if (Kind == clang::OO_Arrow) {
      PrintExpr(Node->getArg(0));
    } else if (Kind == clang::OO_Call) {
      PrintExpr(Node->getArg(0));
      os << '(';
      for (unsigned ArgIdx = 1; ArgIdx < Node->getNumArgs(); ++ArgIdx) {
        if (ArgIdx > 1)
          os << ", ";
        if (!clang::isa<clang::CXXDefaultArgExpr>(Node->getArg(ArgIdx)))
          PrintExpr(Node->getArg(ArgIdx));
      }
      os << ')';
    } else if (Kind == clang::OO_Subscript) {
      PrintExpr(Node->getArg(0));
      os << '[';
      PrintExpr(Node->getArg(1));
      os << ']';
    } else if (Node->getNumArgs() == 1) {
      os << OpStrings[Kind] << ' ';
      PrintExpr(Node->getArg(0));
    } else if (Node->getNumArgs() == 2) {
      PrintExpr(Node->getArg(0));
      os << ' ' << OpStrings[Kind] << ' ';
      PrintExpr(Node->getArg(1));
    } else {
      llvm_unreachable("unknown overloaded operator");
    }
  }

  void VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *Node) {
    // If we have a conversion operator call only print the argument.
    auto *MD = Node->getMethodDecl();
    if (MD && clang::isa<clang::CXXConversionDecl>(MD)) {
      PrintExpr(Node->getImplicitObjectArgument());
      return;
    }
    VisitCallExpr(clang::cast<clang::CallExpr>(Node));
  }

  void VisitCXXNamedCastExpr(clang::CXXNamedCastExpr *Node) {
    os << Node->getCastName() << '<';
    Node->getTypeAsWritten().print(os, Policy);
    os << ">(";
    PrintExpr(Node->getSubExpr());
    os << ')';
  }

  void VisitCXXStaticCastExpr(clang::CXXStaticCastExpr *Node) {
    VisitCXXNamedCastExpr(Node);
  }

  void VisitCXXDynamicCastExpr(clang::CXXDynamicCastExpr *Node) {
    VisitCXXNamedCastExpr(Node);
  }

  void VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *Node) {
    VisitCXXNamedCastExpr(Node);
  }

  void VisitCXXConstCastExpr(clang::CXXConstCastExpr *Node) {
    VisitCXXNamedCastExpr(Node);
  }

  void VisitCXXTypeidExpr(clang::CXXTypeidExpr *Node) {
    os << "typeid(";
    if (Node->isTypeOperand()) {
      Node->getTypeOperandSourceInfo()->getType().print(os, Policy);
    } else {
      PrintExpr(Node->getExprOperand());
    }
    os << ')';
  }

  void VisitCXXUuidofExpr(clang::CXXUuidofExpr *Node) {
    os << "__uuidof(";
    if (Node->isTypeOperand()) {
      Node->getTypeOperandSourceInfo()->getType().print(os, Policy);
    } else {
      PrintExpr(Node->getExprOperand());
    }
    os << ')';
  }

  void VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *Node) {
    PrintExpr(Node->getBaseExpr());
    if (Node->isArrow())
      os << "->";
    else
      os << '.';
    if (auto *Qualifier =
        Node->getQualifierLoc().getNestedNameSpecifier())
      Qualifier->print(os, Policy);
    os << Node->getPropertyDecl()->getDeclName();
  }

  void VisitMSPropertySubscriptExpr(clang::MSPropertySubscriptExpr *Node) {
    PrintExpr(Node->getBase());
    os << '[';
    PrintExpr(Node->getIdx());
    os << ']';
  }

  void VisitUserDefinedLiteral(clang::UserDefinedLiteral *Node) {
    switch (Node->getLiteralOperatorKind()) {
    case clang::UserDefinedLiteral::LOK_Raw:
      os << clang::cast<clang::StringLiteral>(Node->getArg(0)->IgnoreImpCasts())->getString();
      break;
    case clang::UserDefinedLiteral::LOK_Template: {
      auto *DRE = clang::cast<clang::DeclRefExpr>(Node->getCallee()->IgnoreImpCasts());
      const auto *Args =
        clang::cast<clang::FunctionDecl>(DRE->getDecl())->getTemplateSpecializationArgs();
      assert(Args);

      if (Args->size() != 1) {
        os << "operator\"\"" << Node->getUDSuffix()->getName();
        clang::printTemplateArgumentList(os, Args->asArray(), Policy);
        os << "()";
        return;
      }

      const auto &Pack = Args->get(0);
      for (const auto &P : Pack.pack_elements()) {
        char C = (char)P.getAsIntegral().getZExtValue();
        os << C;
      }
      break;
    }
    case clang::UserDefinedLiteral::LOK_Integer: {
      // Print integer literal without suffix.
      auto *Int = clang::cast<clang::IntegerLiteral>(Node->getCookedLiteral());
      os << Int->getValue().toString(10, /*isSigned*/false);
      break;
    }
    case clang::UserDefinedLiteral::LOK_Floating: {
      // Print floating literal without suffix.
      auto *Float = clang::cast<clang::FloatingLiteral>(Node->getCookedLiteral());
      PrintFloatingLiteral(os, Float, /*PrintSuffix=*/false);
      break;
    }
    case clang::UserDefinedLiteral::LOK_String:
    case clang::UserDefinedLiteral::LOK_Character:
      PrintExpr(Node->getCookedLiteral());
      break;
    }
    os << Node->getUDSuffix()->getName();
  }

  void VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *Node) {
    os << (Node->getValue() ? "true" : "false");
  }

  void VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr*) {
    os << "nullptr";
  }

  void VisitCXXThisExpr(clang::CXXThisExpr*) {
    os << "this";
  }

  void VisitCXXThrowExpr(clang::CXXThrowExpr *Node) {
    if (!Node->getSubExpr())
      os << "throw";
    else {
      os << "throw ";
      PrintExpr(Node->getSubExpr());
    }
  }

  void VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr*) {
    // Nothing to print: we picked up the default argument.
  }

  void VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr*) {
    // Nothing to print: we picked up the default initializer.
  }

  void VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *Node) {
    Node->getType().print(os, Policy);
    // If there are no parens, this is list-initialization, and the braces are
    // part of the syntax of the inner construct.
    if (Node->getLParenLoc().isValid())
      os << '(';
    PrintExpr(Node->getSubExpr());
    if (Node->getLParenLoc().isValid())
      os << ')';
  }

  void VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *Node) {
    PrintExpr(Node->getSubExpr());
  }

  void VisitCXXTemporaryObjectExpr(clang::CXXTemporaryObjectExpr *Node) {
    Node->getType().print(os, Policy);
    if (Node->isStdInitListInitialization())
      /* Nothing to do; braces are part of creating the std::initializer_list. */;
    else if (Node->isListInitialization())
      os << '{';
    else
      os << '(';
    for (auto Arg = Node->arg_begin(), ArgEnd = Node->arg_end();
         Arg != ArgEnd; ++Arg) {
      if ((*Arg)->isDefaultArgument())
        break;
      if (Arg != Node->arg_begin())
        os << ", ";
      PrintExpr(*Arg);
    }
    if (Node->isStdInitListInitialization())
      /* See above. */;
    else if (Node->isListInitialization())
      os << '}';
    else
      os << ')';
  }

  void VisitLambdaExpr(clang::LambdaExpr *Node) {
    os << '[';
    bool NeedComma = false;
    switch (Node->getCaptureDefault()) {
    case clang::LCD_None:
      break;

    case clang::LCD_ByCopy:
      os << '=';
      NeedComma = true;
      break;

    case clang::LCD_ByRef:
      os << '&';
      NeedComma = true;
      break;
    }
    for (auto&& C : Node->explicit_captures()) {
      if (NeedComma)
        os << ", ";
      NeedComma = true;

      switch (C.getCaptureKind()) {
      case clang::LCK_This:
        os << "this";
        break;
      case clang::LCK_StarThis:
        os << "*this";
        break;
      case clang::LCK_ByRef:
        if (Node->getCaptureDefault() != clang::LCD_ByRef || Node->isInitCapture(&C))
          os << '&';
        os << C.getCapturedVar()->getName();
        break;

      case clang::LCK_ByCopy:
        os << C.getCapturedVar()->getName();
        break;
      case clang::LCK_VLAType:
        llvm_unreachable("VLA type in explicit captures.");
      }

      if (Node->isInitCapture(&C))
        PrintExpr(C.getCapturedVar()->getInit());
    }
    os << ']';

    if (Node->hasExplicitParameters()) {
      os << " (";
      auto *Method = Node->getCallOperator();
      NeedComma = false;
      for (auto P : Method->parameters()) {
        if (NeedComma) {
          os << ", ";
        } else {
          NeedComma = true;
        }
        std::string ParamStr = P->getNameAsString();
        P->getOriginalType().print(os, Policy, ParamStr);
      }
      if (Method->isVariadic()) {
        if (NeedComma)
          os << ", ";
        os << "...";
      }
      os << ')';

      if (Node->isMutable())
        os << " mutable";

      const auto *Proto
        = Method->getType()->getAs<clang::FunctionProtoType>();
      Proto->printExceptionSpecification(os, Policy);

      // FIXME: Attributes

      // Print the trailing return type if it was specified in the source.
      if (Node->hasExplicitResultType()) {
        os << " -> ";
        Proto->getReturnType().print(os, Policy);
      }
    }

    // Print the body.
    auto *Body = Node->getBody();
    os << ' ';
    PrintStmt(Body);
  }

  void VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *Node) {
    if (auto *TSInfo = Node->getTypeSourceInfo())
      TSInfo->getType().print(os, Policy);
    else
      Node->getType().print(os, Policy);
    os << "()";
  }

  void VisitCXXNewExpr(clang::CXXNewExpr *E) {
    if (E->isGlobalNew())
      os << "::";
    os << "new ";
    unsigned NumPlace = E->getNumPlacementArgs();
    if (NumPlace > 0 && !clang::isa<clang::CXXDefaultArgExpr>(E->getPlacementArg(0))) {
      os << '(';
      PrintExpr(E->getPlacementArg(0));
      for (unsigned i = 1; i < NumPlace; ++i) {
        if (clang::isa<clang::CXXDefaultArgExpr>(E->getPlacementArg(i)))
          break;
        os << ", ";
        PrintExpr(E->getPlacementArg(i));
      }
      os << ") ";
    }
    if (E->isParenTypeId())
      os << '(';
    std::string TypeS;
    if (clang::Expr *Size = E->getArraySize()) {
      llvm::raw_string_ostream s(TypeS);
      s << '[';
      Visit(Size);
      s << ']';
    }
    E->getAllocatedType().print(os, Policy, TypeS);
    if (E->isParenTypeId())
      os << ')';

    auto InitStyle = E->getInitializationStyle();
    if (InitStyle) {
      if (InitStyle == clang::CXXNewExpr::CallInit)
        os << '(';
      PrintExpr(E->getInitializer());
      if (InitStyle == clang::CXXNewExpr::CallInit)
        os << ')';
    }
  }

  void VisitCXXDeleteExpr(clang::CXXDeleteExpr *E) {
    if (E->isGlobalDelete())
      os << "::";
    os << "delete ";
    if (E->isArrayForm())
      os << "[] ";
    PrintExpr(E->getArgument());
  }

  void VisitCXXPseudoDestructorExpr(clang::CXXPseudoDestructorExpr *E) {
    PrintExpr(E->getBase());
    if (E->isArrow())
      os << "->";
    else
      os << '.';
    if (E->getQualifier())
      E->getQualifier()->print(os, Policy);
    os << '~';

    if (auto *II = E->getDestroyedTypeIdentifier())
      os << II->getName();
    else
      E->getDestroyedType().print(os, Policy);
  }

  void VisitCXXConstructExpr(clang::CXXConstructExpr *E) {
    if (E->isListInitialization() && !E->isStdInitListInitialization())
      os << '{';

    for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
      if (clang::isa<clang::CXXDefaultArgExpr>(E->getArg(i))) {
        // Don't print any defaulted arguments
        break;
      }

      if (i) os << ", ";
      PrintExpr(E->getArg(i));
    }

    if (E->isListInitialization() && !E->isStdInitListInitialization())
      os << '}';
  }

  void VisitCXXInheritedCtorInitExpr(clang::CXXInheritedCtorInitExpr*) {
    // Parens are printed by the surrounding context.
    os << "<forwarded>";
  }

  void VisitCXXStdInitializerListExpr(clang::CXXStdInitializerListExpr *E) {
    PrintExpr(E->getSubExpr());
  }

  void VisitExprWithCleanups(clang::ExprWithCleanups *E) {
    // Just forward to the subexpression.
    PrintExpr(E->getSubExpr());
  }

  void VisitCXXUnresolvedConstructExpr(clang::CXXUnresolvedConstructExpr *Node) {
    Node->getTypeAsWritten().print(os, Policy);
    os << '(';
    for (auto Arg = Node->arg_begin(), ArgEnd = Node->arg_end();
         Arg != ArgEnd; ++Arg) {
      if (Arg != Node->arg_begin())
        os << ", ";
      PrintExpr(*Arg);
    }
    os << ')';
  }

  void VisitCXXDependentScopeMemberExpr(clang::CXXDependentScopeMemberExpr *Node) {
    if (!Node->isImplicitAccess()) {
      PrintExpr(Node->getBase());
      if(Node->isArrow())
        os << "->";
      else
        os << '.';
    }
    if (auto *Qualifier = Node->getQualifier())
      Qualifier->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getMemberNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }

  void VisitUnresolvedMemberExpr(clang::UnresolvedMemberExpr *Node) {
    if (!Node->isImplicitAccess()) {
      PrintExpr(Node->getBase());
      if(Node->isArrow())
        os << "->";
      else
        os << '.';
    }
    if (auto *Qualifier = Node->getQualifier())
      Qualifier->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getMemberNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }

  static const char *getTypeTraitName(clang::TypeTrait TT) {
    switch (TT) {
#define TYPE_TRAIT_1(Spelling, Name, Key) \
  case clang::UTT_##Name: return #Spelling;
#define TYPE_TRAIT_2(Spelling, Name, Key) \
  case clang::BTT_##Name: return #Spelling;
#define TYPE_TRAIT_N(Spelling, Name, Key) \
    case clang::TT_##Name: return #Spelling;
#include "clang/Basic/TokenKinds.def"
#undef TYPE_TRAIT_N
#undef TYPE_TRAIT_2
#undef TYPE_TRAIT_1
    }
    llvm_unreachable("Type trait not covered by switch");
  }

  static const char *getTypeTraitName(clang::ArrayTypeTrait ATT) {
    switch (ATT) {
    case clang::ATT_ArrayRank:        return "__array_rank";
    case clang::ATT_ArrayExtent:      return "__array_extent";
    }
    llvm_unreachable("Array type trait not covered by switch");
  }

  static const char *getExpressionTraitName(clang::ExpressionTrait ET) {
    switch (ET) {
    case clang::ET_IsLValueExpr:      return "__is_lvalue_expr";
    case clang::ET_IsRValueExpr:      return "__is_rvalue_expr";
    }
    llvm_unreachable("Expression type trait not covered by switch");
  }

  void VisitTypeTraitExpr(clang::TypeTraitExpr *E) {
    os << getTypeTraitName(E->getTrait()) << '(';
    for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I) {
      if (I > 0)
        os << ", ";
      E->getArg(I)->getType().print(os, Policy);
    }
    os << ')';
  }

  void VisitArrayTypeTraitExpr(clang::ArrayTypeTraitExpr *E) {
    os << getTypeTraitName(E->getTrait()) << '(';
    E->getQueriedType().print(os, Policy);
    os << ')';
  }

  void VisitExpressionTraitExpr(clang::ExpressionTraitExpr *E) {
    os << getExpressionTraitName(E->getTrait()) << '(';
    PrintExpr(E->getQueriedExpression());
    os << ')';
  }

  void VisitCXXNoexceptExpr(clang::CXXNoexceptExpr *E) {
    os << "noexcept(";
    PrintExpr(E->getOperand());
    os << ')';
  }

  void VisitPackExpansionExpr(clang::PackExpansionExpr *E) {
    PrintExpr(E->getPattern());
    os << "...";
  }

  void VisitSizeOfPackExpr(clang::SizeOfPackExpr *E) {
    os << "sizeof...(" << *E->getPack() << ')';
  }

  void VisitSubstNonTypeTemplateParmPackExpr(clang::SubstNonTypeTemplateParmPackExpr *Node) {
    os << *Node->getParameterPack();
  }

  void VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *Node) {
    Visit(Node->getReplacement());
  }

  void VisitFunctionParmPackExpr(clang::FunctionParmPackExpr *E) {
    os << *E->getParameterPack();
  }

  void VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *Node){
    PrintExpr(Node->GetTemporaryExpr());
  }

  void VisitCXXFoldExpr(clang::CXXFoldExpr *E) {
    os << '(';
    if (E->getLHS()) {
      PrintExpr(E->getLHS());
      os << ' ' << clang::BinaryOperator::getOpcodeStr(E->getOperator()) << ' ';
    }
    os << "...";
    if (E->getRHS()) {
      os << ' ' << clang::BinaryOperator::getOpcodeStr(E->getOperator()) << ' ';
      PrintExpr(E->getRHS());
    }
    os << ')';
  }

  // C++ Coroutines TS

  void VisitCoroutineBodyStmt(clang::CoroutineBodyStmt *S) {
    Visit(S->getBody());
  }

  void VisitCoreturnStmt(clang::CoreturnStmt *S) {
    os << "co_return";
    if (S->getOperand()) {
      os << ' ';
      Visit(S->getOperand());
    }
    os << ';';
  }

  void VisitCoawaitExpr(clang::CoawaitExpr *S) {
    os << "co_await ";
    PrintExpr(S->getOperand());
  }


/*  void VisitDependentCoawaitExpr(clang::DependentCoawaitExpr *S) {
    os << "co_await ";
    PrintExpr(S->getOperand());
  }*/


  void VisitCoyieldExpr(clang::CoyieldExpr *S) {
    os << "co_yield ";
    PrintExpr(S->getOperand());
  }

  void VisitBlockExpr(clang::BlockExpr *Node) {
    auto *BD = Node->getBlockDecl();
    os << '^';

    const auto *AFT = Node->getFunctionType();

    if (clang::isa<clang::FunctionNoProtoType>(AFT)) {
      os << "()";
    } else if (!BD->param_empty() || clang::cast<clang::FunctionProtoType>(AFT)->isVariadic()) {
      os << '(';
      for (auto AI = BD->param_begin(),
           E = BD->param_end(); AI != E; ++AI) {
        if (AI != BD->param_begin()) os << ", ";
        std::string ParamStr = (*AI)->getNameAsString();
        (*AI)->getType().print(os, Policy, ParamStr);
      }

      const auto *FT = clang::cast<clang::FunctionProtoType>(AFT);
      if (FT->isVariadic()) {
        if (!BD->param_empty()) os << ", ";
        os << "...";
      }
      os << ')';
    }
    os << "{ }";
  }

  void VisitOpaqueValueExpr(clang::OpaqueValueExpr *Node) { 
    PrintExpr(Node->getSourceExpr());
  }

  void VisitTypoExpr(clang::TypoExpr*) {
    // TODO: Print something reasonable for a TypoExpr, if necessary.
    llvm_unreachable("Cannot print TypoExpr nodes");
  }

  void VisitAsTypeExpr(clang::AsTypeExpr *Node) {
    os << "__builtin_astype(";
    PrintExpr(Node->getSrcExpr());
    os << ", ";
    Node->getType().print(os, Policy);
    os << ')';
  }
};

class decl_visitor : public clang::DeclVisitor<decl_visitor>{
  llvm::raw_ostream& indent() { return indent(indentation); }
  static bool has_annotation(clang::Decl* decl, llvm::StringRef s){
    if(decl->hasAttrs())
      for(auto&& x : decl->getAttrs())
        if(auto a = clang::dyn_cast<clang::AnnotateAttr>(x))
          if(s == a->getAnnotation())
            return true;
    return false;
  }
  template<typename T>
  static clang::CXXRecordDecl* get_unnamed_record_decl(T* f){
    const clang::Type* t = f->getType().split().Ty;
    if(t->getTypeClass() == clang::Type::Elaborated){
      auto r = clang::cast<clang::ElaboratedType>(t)->getNamedType()->getAsCXXRecordDecl();
      if(r && !r->getIdentifier())
        return r;
    }
    return nullptr;
  }
  template<typename T>
  static clang::EnumDecl* get_unnamed_enum_decl(T* f){
    const clang::Type* t = f->getType().split().Ty;
    if(t->getTypeClass() == clang::Type::Elaborated){
      auto r_ = clang::cast<clang::ElaboratedType>(t)->getNamedType()->getAsTagDecl();
      auto r = clang::dyn_cast<clang::EnumDecl>(r_);
      if(r && !r->getIdentifier())
        return r;
    }
    return nullptr;
  }

  ostreams os;
  clang::PrintingPolicy policy;
  unsigned indentation;
  bool PrintInstantiation;
  stmt_visitor sv;
  int print_out_counter = 0;

public:
  decl_visitor(llvm::raw_ostream& os, const clang::PrintingPolicy& policy,
               unsigned indentation = 0, bool PrintInstantiation = false)
    : os(os), policy(policy), indentation(indentation),
      PrintInstantiation(PrintInstantiation),
      sv{this->os, this->policy, this->indentation, *this} { }

  llvm::raw_ostream& indent(unsigned indentation) {
    for (unsigned i = 0; i != indentation; ++i)
      os << "  ";
    return os;
  }

  template<typename T>
  void prettyPrintAttributes(T* D){
    if (policy.PolishForDeclaration)
      return;

    for (auto *x : D->getAttrs()) {
      if (x->isInherited() || x->isImplicit())
        continue;
      switch (x->getKind()) {
#define ATTR(X)
#define PRAGMA_SPELLING_ATTR(X) case clang::attr::X:
#include "clang/Basic/AttrList.inc"
#undef PRAGMA_SPELLING_ATTR
#undef ATTR
        break;
      default:
        x->printPretty(os, policy);
        break;
      }
    }
  }
  void prettyPrintPragmas(clang::Decl* D){
    if (policy.PolishForDeclaration)
      return;

    for (auto *x : D->getAttrs()) {
      switch (x->getKind()) {
#define ATTR(X)
#define PRAGMA_SPELLING_ATTR(X) case clang::attr::X:
#include "clang/Basic/AttrList.inc"
#undef PRAGMA_SPELLING_ATTR
#undef ATTR
        x->printPretty(os, policy);
        indent();
        break;
      default:
        break;
      }
    }
  }

  void printGroup(clang::Decl** Begin, unsigned NumDecls) {
    if (NumDecls == 1) {
      Visit(*Begin);
      return;
    }
    clang::Decl** End = Begin + NumDecls;
    auto backup = policy;
    auto* TD = clang::dyn_cast<clang::TagDecl>(*Begin);
    if (TD)
      ++Begin;

    bool isFirst = true;
    for ( ; Begin != End; ++Begin) {
      if (isFirst){
        if(TD)
          policy.IncludeTagDefinition = true;
        policy.SuppressSpecifiers = false;
        isFirst = false;
      }
      else{
        os << ", ";
        policy.IncludeTagDefinition = false;
        policy.SuppressSpecifiers = true;
      }

      Visit(*Begin);
    }
    policy = backup;
  }

  void printDeclType(clang::QualType T, llvm::StringRef DeclName, bool Pack = false) {
    // Normally, a PackExpansionType is written as T[3]... (for instance, as a
    // template argument), but if it is the type of a declaration, the ellipsis
    // is placed before the name being declared.
    if (auto PET = T->getAs<clang::PackExpansionType>()) {
      Pack = true;
      T = PET->getPattern();
    }
    T.print(os, policy, (Pack ? "..." : "") + DeclName, indentation);
  }

  void ProcessDeclGroup(llvm::SmallVectorImpl<clang::Decl*>& Decls) {
    this->indent();
    printGroup(Decls.data(), Decls.size());
    os << ";\n";
    Decls.clear();

  }

  void Print(clang::AccessSpecifier AS) {
    switch(AS) {
    case clang::AS_none:      llvm_unreachable("No access specifier!");
    case clang::AS_public:    os << "public"; break;
    case clang::AS_protected: os << "protected"; break;
    case clang::AS_private:   os << "private"; break;
    }
  }

  void VisitDeclContext(clang::DeclContext *DC, bool indent = true) {
    if (policy.TerseOutput)
      return;

    if (indent)
      indentation += policy.Indentation;

    llvm::SmallVector<clang::Decl*, 2> Decls;
    std::string os_source;
    llvm::raw_string_ostream ros(os_source);
    for(auto b = DC->decls_begin(), e = DC->decls_end(); b != e; ++b){
      auto&& x = *b;

      // Don't print ObjCIvarDecls, as they are printed when visiting the
      // containing ObjCInterfaceDecl.
      if (clang::isa<clang::ObjCIvarDecl>(x))
        continue;

      // Skip over implicit declarations in pretty-printing mode.
      if (x->isImplicit())
        continue;

      // Don't print implicit specializations, as they are printed when visiting
      // corresponding templates.
      if (auto FD = clang::dyn_cast<clang::FunctionDecl>(x)){
        if (FD->getTemplateSpecializationKind() == clang::TSK_ImplicitInstantiation &&
            !clang::isa<clang::ClassTemplateSpecializationDecl>(DC))
          continue;
        if ( FD->getStorageClass() == clang::SC_Static
          && FD->getReturnType().getAsString() == "void"
          && FD->getQualifiedNameAsString() == "__clpy_begin_print_out"
          && has_annotation(FD, "clpy_begin_print_out")
          && FD->param_size() == 0
          && FD->hasBody() == false){
          ++print_out_counter;
          continue;
        }
        else if ( FD->getStorageClass() == clang::SC_Static
               && FD->getReturnType().getAsString() == "void"
               && FD->getQualifiedNameAsString() == "__clpy_end_print_out"
               && has_annotation(FD, "clpy_end_print_out")
               && FD->param_size() == 0
               && FD->hasBody() == false){
          --print_out_counter;
          continue;
        }
      }

      if(print_out_counter <= 0)
        continue;

      ros.flush();
      os << os_source;
      os_source.clear();
      auto _ = os.scoped_push(ros);

      // The next bits of code handles stuff like "struct {int x;} a,b"; we're
      // forced to merge the declarations because there's no other way to
      // refer to the struct in question.  This limited merging is safe without
      // a bunch of other checks because it only merges declarations directly
      // referring to the tag, not typedefs.
      //
      // Check whether the current declaration should be grouped with a previous
      // unnamed struct.
      auto cur_decl_type = clang::QualType{};
      if(auto tdnd = clang::dyn_cast<clang::TypedefNameDecl>(x))
        cur_decl_type = tdnd->getUnderlyingType();
      else if(auto vd = clang::dyn_cast<clang::ValueDecl>(x))
        cur_decl_type = vd->getType();
      if(!Decls.empty() && !cur_decl_type.isNull()){
        clang::QualType base_type = cur_decl_type;
        while(!base_type->isSpecifierType()){
          if(clang::isa<clang::TypedefType>(base_type))
            break;
          else if (auto pt = base_type->getAs<clang::PointerType>())
            base_type = pt->getPointeeType();
          else if (auto bpt = base_type->getAs<clang::BlockPointerType>())
            base_type = bpt->getPointeeType();
          else if (auto at = clang::dyn_cast<clang::ArrayType>(base_type))
            base_type = at->getElementType();
          else if (auto ft = base_type->getAs<clang::FunctionType>())
            base_type = ft->getReturnType();
          else if (auto vt = base_type->getAs<clang::VectorType>())
            base_type = vt->getElementType();
          else if (auto rt = base_type->getAs<clang::ReferenceType>())
            base_type = rt->getPointeeType();
          else if (auto at = base_type->getAs<clang::AutoType>())
            base_type = at->getDeducedType();
          else
            llvm_unreachable("Unknown declarator!");
        }
        if(!base_type.isNull() && clang::isa<clang::ElaboratedType>(base_type))
          base_type = clang::cast<clang::ElaboratedType>(base_type)->getNamedType();
        if (!base_type.isNull() && clang::isa<clang::TagType>(base_type) &&
            clang::cast<clang::TagType>(base_type)->getDecl() == Decls[0]) {
          Decls.push_back(x);
          continue;
        }
      }

      // If we have a merged group waiting to be handled, handle it now.
      if (!Decls.empty())
        ProcessDeclGroup(Decls);

      // If the current declaration is an unnamed tag type, save it
      // so we can merge it with the subsequent declaration(s) using it.
      if (clang::isa<clang::TagDecl>(x) && !clang::cast<clang::TagDecl>(x)->getIdentifier()) {
        Decls.push_back(x);
        continue;
      }

      if (clang::isa<clang::AccessSpecDecl>(x)) {
        indentation -= policy.Indentation;
        this->indent();
        Print(x->getAccess());
        os << ":\n";
        indentation += policy.Indentation;
        continue;
      }

      this->indent();
      Visit(x);

      // FIXME: Need to be able to tell the DeclPrinter when
      const char *Terminator = nullptr;
      if (clang::isa<clang::OMPThreadPrivateDecl>(x) || clang::isa<clang::OMPDeclareReductionDecl>(x))
        Terminator = nullptr;
      else if (clang::isa<clang::ObjCMethodDecl>(x) && clang::cast<clang::ObjCMethodDecl>(x)->hasBody())
        Terminator = nullptr;
      else if (auto FD = clang::dyn_cast<clang::FunctionDecl>(x)) {
        const bool special_definition = FD->isPure() || FD->isDefaulted() || FD->isDeleted();
        if(special_definition)
          Terminator = ";\n";
        else if (FD->isThisDeclarationADefinition())
          Terminator = nullptr;
        else
          Terminator = ";";
      } else if (auto TD = clang::dyn_cast<clang::FunctionTemplateDecl>(x)) {
        const bool special_definition = TD->getTemplatedDecl()->isPure() || TD->getTemplatedDecl()->isDefaulted() || TD->getTemplatedDecl()->isDeleted();
        if(special_definition)
          Terminator = ";\n";
        else if (TD->getTemplatedDecl()->isThisDeclarationADefinition() && !special_definition)
          Terminator = nullptr;
        else
          Terminator = ";";
      } else if (clang::isa<clang::NamespaceDecl>(x) || clang::isa<clang::LinkageSpecDecl>(x) ||
               clang::isa<clang::ObjCImplementationDecl>(x) ||
               clang::isa<clang::ObjCInterfaceDecl>(x) ||
               clang::isa<clang::ObjCProtocolDecl>(x) ||
               clang::isa<clang::ObjCCategoryImplDecl>(x) ||
               clang::isa<clang::ObjCCategoryDecl>(x))
        Terminator = nullptr;
      else if (clang::isa<clang::EnumConstantDecl>(x)) {
        if(std::next(b) != e)
          Terminator = ",";
      } else
        Terminator = ";";

      if (Terminator)
        os << Terminator;
      if (!policy.TerseOutput &&
          ((clang::isa<clang::FunctionDecl>(x) &&
            clang::cast<clang::FunctionDecl>(x)->doesThisDeclarationHaveABody()) ||
           (clang::isa<clang::FunctionTemplateDecl>(x) &&
            clang::cast<clang::FunctionTemplateDecl>(x)->getTemplatedDecl()->doesThisDeclarationHaveABody())))
        ; // StmtPrinter already added '\n' after CompoundStmt.
      else
        os << '\n';

    }

    ros.flush();
    os << os_source;

    if (!Decls.empty())
      ProcessDeclGroup(Decls);

    if (indent)
      indentation -= policy.Indentation;
  }

  void VisitTranslationUnitDecl(clang::TranslationUnitDecl *D) {
    VisitDeclContext(D, false);
  }

  void VisitTypedefDecl(clang::TypedefDecl *D) {
    if (!policy.SuppressSpecifiers) {
      os << "typedef ";
      
      if (D->isModulePrivate())
        os << "__module_private__ ";
    }
    if(auto r = get_unnamed_record_decl(D->getTypeSourceInfo())){
      VisitCXXRecordDecl(r, true);
      os << D->getName();
    }
    else if(auto r = get_unnamed_enum_decl(D->getTypeSourceInfo())){
      VisitEnumDecl(r, true);
      os << D->getName();
    }
    else{
      policy.SuppressTagKeyword = 1;
      printDeclType(D->getTypeSourceInfo()->getType(), D->getName());
      policy.SuppressTagKeyword = 0;
    }
    prettyPrintAttributes(D);
  }

  void VisitTypeAliasDecl(clang::TypeAliasDecl *D) {
    os << "using " << *D;
    prettyPrintAttributes(D);
    os << " = ";
    if(auto r = get_unnamed_record_decl(D->getTypeSourceInfo()))
      VisitCXXRecordDecl(r, true);
    else if(auto r = get_unnamed_enum_decl(D->getTypeSourceInfo()))
      VisitEnumDecl(r, true);
    else
      os << D->getTypeSourceInfo()->getType().getAsString();
  }

  void VisitEnumDecl(clang::EnumDecl *D, bool force = false) {
    if(!D->isCompleteDefinition() || (!D->getIdentifier() && !force) || policy.SuppressSpecifiers)
      return;
    if (!policy.SuppressSpecifiers && D->isModulePrivate())
      os << "__module_private__ ";
    os << "enum ";
    if (D->isScoped()) {
      if (D->isScopedUsingClassTag())
        os << "class ";
      else
        os << "struct ";
    }
    os << *D;

    if (D->isFixed() && D->getASTContext().getLangOpts().CPlusPlus11)
      os << " : " << D->getIntegerType().stream(policy);

    if (D->isCompleteDefinition()) {
      os << " {\n";
      VisitDeclContext(D);
      indent() << '}';
    }
    prettyPrintAttributes(D);
  }

  void VisitEnumConstantDecl(clang::EnumConstantDecl *D) {
    os << *D;
    prettyPrintAttributes(D);
    if (clang::Expr *Init = D->getInitExpr()) {
      os << " = ";
      sv.Visit(Init);
    }
  }

  void VisitFunctionDecl(clang::FunctionDecl *D) {
    if (!D->getDescribedFunctionTemplate() &&
        !D->isFunctionTemplateSpecialization())
      prettyPrintPragmas(D);

    if (D->isFunctionTemplateSpecialization())
      os << "template<> ";
    else if (!D->getDescribedFunctionTemplate()) {
      for (unsigned I = 0, NumTemplateParams = D->getNumTemplateParameterLists();
           I < NumTemplateParams; ++I)
        printTemplateParameters(D->getTemplateParameterList(I));
    }

    clang::CXXConstructorDecl *CDecl = clang::dyn_cast<clang::CXXConstructorDecl>(D);
    clang::CXXConversionDecl *ConversionDecl = clang::dyn_cast<clang::CXXConversionDecl>(D);
    if (!policy.SuppressSpecifiers) {
      switch (D->getStorageClass()) {
      case clang::SC_None: break;
      case clang::SC_Extern: os << "extern "; break;
      case clang::SC_Static: os << "static "; break;
      case clang::SC_PrivateExtern: os << "__private_extern__ "; break;
      case clang::SC_Auto: case clang::SC_Register:
        llvm_unreachable("invalid for functions");
      }

      if (D->isInlineSpecified())  os << "inline ";
      if (D->isVirtualAsWritten()) os << "virtual ";
      if (D->isModulePrivate())    os << "__module_private__ ";
      if (D->isConstexpr() && !D->isExplicitlyDefaulted()) os << "constexpr ";
      if ((CDecl && CDecl->isExplicitSpecified())
       || (ConversionDecl && ConversionDecl->isExplicitSpecified())
         )
        os << "explicit ";
    }

    std::string Proto;
    if (!policy.SuppressScope) {
      if(auto NS = D->getQualifier()) {
        llvm::raw_string_ostream OS(Proto);
        NS->print(OS, policy);
      }
    }
    Proto += D->getNameInfo().getAsString();
    if(auto TArgs = D->getTemplateSpecializationArgs()) {
      auto backup = policy;
      policy.SuppressSpecifiers = false;
      llvm::raw_string_ostream Pos(Proto);
      auto _ = os.scoped_push(Pos);
      printTemplateArguments(*TArgs);
      policy = backup;
    }

    clang::QualType Ty = D->getType();
    while(auto PT = clang::dyn_cast<clang::ParenType>(Ty)) {
      Proto = '(' + Proto + ')';
      Ty = PT->getInnerType();
    }

    if(auto AFT = Ty->getAs<clang::FunctionType>()) {
      const clang::FunctionProtoType *FT = nullptr;
      if (D->hasWrittenPrototype())
        FT = clang::dyn_cast<clang::FunctionProtoType>(AFT);

      Proto += '(';
      if (FT) {
        auto backup = policy;
        policy.SuppressSpecifiers = false;
        llvm::raw_string_ostream Pos(Proto);
        auto _ = os.scoped_push(Pos);
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i) Pos << ", ";
          VisitParmVarDecl(D->getParamDecl(i));
        }
        backup = policy;

        if (FT->isVariadic()) {
          if (D->getNumParams()) Pos << ", ";
          Pos << "...";
        }
      } else if (D->doesThisDeclarationHaveABody() && !D->hasPrototype()) {
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i)
            Proto += ", ";
          Proto += D->getParamDecl(i)->getNameAsString();
        }
      }

      Proto += ')';
      
      if (FT) {
        if (FT->isConst())
          Proto += " const";
        if (FT->isVolatile())
          Proto += " volatile";
        if (FT->isRestrict())
          Proto += " restrict";

        switch (FT->getRefQualifier()) {
        case clang::RQ_None:
          break;
        case clang::RQ_LValue:
          Proto += " &";
          break;
        case clang::RQ_RValue:
          Proto += " &&";
          break;
        }
      }

      auto subpolicy = policy;
      subpolicy.SuppressSpecifiers = false;
      if (FT && FT->hasDynamicExceptionSpec()) {
        Proto += " throw(";
        if (FT->getExceptionSpecType() == clang::EST_MSAny)
          Proto += "...";
        else 
          for (unsigned I = 0, N = FT->getNumExceptions(); I != N; ++I) {
            if (I)
              Proto += ", ";

            Proto += FT->getExceptionType(I).getAsString(subpolicy);
          }
        Proto += ')';
      } else if (FT && isNoexceptExceptionSpec(FT->getExceptionSpecType())) {
        Proto += " noexcept";
        if (FT->getExceptionSpecType() == clang::EST_ComputedNoexcept) {
          Proto += '(';
          llvm::raw_string_ostream Eos(Proto);
          FT->getNoexceptExpr()->printPretty(Eos, nullptr, subpolicy,
                                             indentation);
          Eos.flush();
          Proto += Eos.str();
          Proto += ')';
        }
      }

      if (CDecl) {
        bool HasInitializerList = false;
        for (const auto *BMInitializer : CDecl->inits()) {
          if (BMInitializer->isInClassMemberInitializer())
            continue;

          if (!HasInitializerList) {
            Proto += " : ";
            os << Proto;
            Proto.clear();
            HasInitializerList = true;
          } else
            os << ", ";

          if (BMInitializer->isAnyMemberInitializer()) {
            clang::FieldDecl *FD = BMInitializer->getAnyMember();
            os << *FD;
          } else {
            os << clang::QualType(BMInitializer->getBaseClass(), 0).getAsString(policy);
          }
          
          os << '(';
          if (!BMInitializer->getInit()) {
            // Nothing to print
          } else {
            clang::Expr *Init = BMInitializer->getInit();
            if (clang::ExprWithCleanups *Tmp = clang::dyn_cast<clang::ExprWithCleanups>(Init))
              Init = Tmp->getSubExpr();
            
            Init = Init->IgnoreParens();

            clang::Expr *SimpleInit = nullptr;
            clang::Expr **Args = nullptr;
            unsigned NumArgs = 0;
            if (clang::ParenListExpr *ParenList = clang::dyn_cast<clang::ParenListExpr>(Init)) {
              Args = ParenList->getExprs();
              NumArgs = ParenList->getNumExprs();
            } else if (clang::CXXConstructExpr *Construct
                                          = clang::dyn_cast<clang::CXXConstructExpr>(Init)) {
              Args = Construct->getArgs();
              NumArgs = Construct->getNumArgs();
            } else
              SimpleInit = Init;
            
            if (SimpleInit)
              sv.Visit(SimpleInit);
            else {
              for (unsigned I = 0; I != NumArgs; ++I) {
                assert(Args[I] != nullptr && "Expected non-null Expr");
                if (clang::isa<clang::CXXDefaultArgExpr>(Args[I]))
                  break;
                
                if (I)
                  os << ", ";
                sv.Visit(Args[I]);
              }
            }
          }
          os << ')';
          if (BMInitializer->isPackExpansion())
            os << "...";
        }
      } else if (!ConversionDecl && !clang::isa<clang::CXXDestructorDecl>(D)) {
        if (FT && FT->hasTrailingReturn()) {
          os << Proto << " -> ";
          Proto.clear();
        }
        AFT->getReturnType().print(os, policy, Proto);
        Proto.clear();
      }
      os << Proto;
    } else {
      Ty.print(os, policy, Proto);
    }

    prettyPrintAttributes(D);

    if (D->isPure())
      os << " = 0";
    else if (D->isDeletedAsWritten())
      os << " = delete";
    else if (D->isExplicitlyDefaulted())
      os << " = default";
    else if (D->doesThisDeclarationHaveABody()) {
      if (!policy.TerseOutput) {
        if (!D->hasPrototype() && D->getNumParams()) {
          // This is a K&R function definition, so we need to print the
          // parameters.
          os << '\n';
          auto backup = policy;
          policy.SuppressSpecifiers = false;
          indentation += policy.Indentation;
          for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
            indent();
            VisitParmVarDecl(D->getParamDecl(i));
            os << ";";
          }
          policy = backup;
          indentation -= policy.Indentation;
        } else
          os << ' ';

        if (D->getBody()){
          os << '\n';
          sv.Visit(D->getBody());
        }
      } else {
        if (clang::isa<clang::CXXConstructorDecl>(*D))
          os << " {}";
      }
    }
  }

  void VisitFriendDecl(clang::FriendDecl *D) {
    if (clang::TypeSourceInfo *TSI = D->getFriendType()) {
      unsigned NumTPLists = D->getFriendTypeNumTemplateParameterLists();
      for (unsigned i = 0; i < NumTPLists; ++i)
        printTemplateParameters(D->getFriendTypeTemplateParameterList(i));
      os << "friend ";
      os << ' ' << TSI->getType().getAsString(policy);
    }
    else if (clang::FunctionDecl *FD =
        clang::dyn_cast<clang::FunctionDecl>(D->getFriendDecl())) {
      os << "friend ";
      VisitFunctionDecl(FD);
    }
    else if (clang::FunctionTemplateDecl *FTD =
             clang::dyn_cast<clang::FunctionTemplateDecl>(D->getFriendDecl())) {
      os << "friend ";
      VisitFunctionTemplateDecl(FTD);
    }
    else if (clang::ClassTemplateDecl *CTD =
             clang::dyn_cast<clang::ClassTemplateDecl>(D->getFriendDecl())) {
      os << "friend ";
      VisitRedeclarableTemplateDecl(CTD);
    }
  }

  void VisitFieldDecl(clang::FieldDecl *D) {
    // FIXME: add printing of pragma attributes if required.
    if (!policy.SuppressSpecifiers && D->isMutable())
      os << "mutable ";
    if (!policy.SuppressSpecifiers && D->isModulePrivate())
      os << "__module_private__ ";

    if(auto r = get_unnamed_record_decl(D)){
      VisitCXXRecordDecl(r, true);
      os << ' ' << D->getName();
    }
    else if(auto r = get_unnamed_enum_decl(D->getTypeSourceInfo())){
      VisitEnumDecl(r, true);
      os << ' ' << D->getName();
    }
    else
      printDeclType(D->getType(), D->getName());

    if (D->isBitField()) {
      os << " : ";
      sv.Visit(D->getBitWidth());
    }

    clang::Expr *Init = D->getInClassInitializer();
    if (!policy.SuppressInitializers && Init) {
      if (D->getInClassInitStyle() == clang::ICIS_ListInit)
        os << ' ';
      else
        os << " = ";
      sv.Visit(Init);
    }
    prettyPrintAttributes(D);
  }

  void VisitLabelDecl(clang::LabelDecl *D) {
    os << *D << ':';
  }

  void VisitVarDecl(clang::VarDecl *D) {
    prettyPrintPragmas(D);

    clang::QualType T = D->getTypeSourceInfo()
      ? D->getTypeSourceInfo()->getType()
      : D->getASTContext().getUnqualifiedObjCPointerType(D->getType());

    if (!policy.SuppressSpecifiers) {
      clang::StorageClass SC = D->getStorageClass();
      if (SC != clang::SC_None)
        os << clang::VarDecl::getStorageClassSpecifierString(SC) << ' ';

      switch (D->getTSCSpec()) {
      case clang::TSCS_unspecified:
        break;
      case clang::TSCS___thread:
        os << "__thread ";
        break;
      case clang::TSCS__Thread_local:
        os << "_Thread_local ";
        break;
      case clang::TSCS_thread_local:
        os << "thread_local ";
        break;
      }

      if (D->isModulePrivate())
        os << "__module_private__ ";

      if (D->isConstexpr()) {
        os << "constexpr ";
        T.removeLocalConst();
      }
    }

    if(auto r = get_unnamed_record_decl(D)){
      VisitCXXRecordDecl(r, true);
      os << ' ' << D->getName();
    }
    else if(auto r = get_unnamed_enum_decl(D->getTypeSourceInfo())){
      VisitEnumDecl(r, true);
      os << ' ' << D->getName();
    }
    else
      printDeclType(T, D->getName());
    clang::Expr *Init = D->getInit();
    if (!policy.SuppressInitializers && Init) {
      bool ImplicitInit = false;
      if (auto *Construct =
              clang::dyn_cast<clang::CXXConstructExpr>(Init->IgnoreImplicit())) {
        if (D->getInitStyle() == clang::VarDecl::CallInit &&
            !Construct->isListInitialization()) {
          ImplicitInit = Construct->getNumArgs() == 0 ||
            Construct->getArg(0)->isDefaultArgument();
        }
      }
      if (!ImplicitInit) {
        if ((D->getInitStyle() == clang::VarDecl::CallInit) && !clang::isa<clang::ParenListExpr>(Init))
          os << '(';
        else if (D->getInitStyle() == clang::VarDecl::CInit) {
          os << " = ";
        }
        auto backup = policy;
        policy.SuppressSpecifiers = false;
        policy.IncludeTagDefinition = false;
        sv.Visit(Init);
        policy = backup;
        if ((D->getInitStyle() == clang::VarDecl::CallInit) && !clang::isa<clang::ParenListExpr>(Init))
          os << ')';
      }
    }
    prettyPrintAttributes(D);
  }

  void VisitParmVarDecl(clang::ParmVarDecl *D) {
    VisitVarDecl(D);
  }

  void VisitFileScopeAsmDecl(clang::FileScopeAsmDecl *D) {
    os << "__asm (";
    sv.Visit(D->getAsmString());
    os << ')';
  }

  void VisitStaticAssertDecl(clang::StaticAssertDecl *D) {
    os << "static_assert(";
    sv.Visit(D->getAssertExpr());
    if (clang::StringLiteral *SL = D->getMessage()) {
      os << ", ";
      sv.Visit(SL);
    }
    os << ')';
  }

  void VisitNamespaceDecl(clang::NamespaceDecl *D) {
    if (D->isInline())
      os << "inline ";
    os << "namespace " << *D << " {\n";
    VisitDeclContext(D);
    indent() << '}';
  }

  void VisitUsingDirectiveDecl(clang::UsingDirectiveDecl *D) {
    os << "using namespace ";
    if (D->getQualifier())
      D->getQualifier()->print(os, policy);
    os << *D->getNominatedNamespaceAsWritten();
  }

  void VisitNamespaceAliasDecl(clang::NamespaceAliasDecl *D) {
    os << "namespace " << *D << " = ";
    if (D->getQualifier())
      D->getQualifier()->print(os, policy);
    os << *D->getAliasedNamespace();
  }

  void VisitEmptyDecl(clang::EmptyDecl *D) {
    prettyPrintAttributes(D);
  }

  void VisitCXXRecordDecl(clang::CXXRecordDecl *D, bool force = false) {
    if(!D->isCompleteDefinition() || (!D->getIdentifier() && !force) || policy.SuppressSpecifiers)
      return;
    if(D->getKind() == clang::Decl::Kind::Enum){
      D->print(os, indentation);
    }
    // FIXME: add printing of pragma attributes if required.
    if (!policy.SuppressSpecifiers && D->isModulePrivate())
      os << "__module_private__ ";
    os << D->getKindName();

    prettyPrintAttributes(D);

    if (D->getIdentifier()) {
      os << ' ' << *D;

      if (auto S = clang::dyn_cast<clang::ClassTemplatePartialSpecializationDecl>(D))
        printTemplateArguments(S->getTemplateArgs(), S->getTemplateParameters());
      else if (auto S = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(D))
        printTemplateArguments(S->getTemplateArgs());
    }

    if (D->isCompleteDefinition()) {
      // Print the base classes
      if (D->getNumBases()) {
        os << " : ";
        for (auto Base = D->bases_begin(),
               BaseEnd = D->bases_end(); Base != BaseEnd; ++Base) {
          if (Base != D->bases_begin())
            os << ", ";

          if (Base->isVirtual())
            os << "virtual ";

          clang::AccessSpecifier AS = Base->getAccessSpecifierAsWritten();
          if (AS != clang::AS_none) {
            Print(AS);
            os << ' ';
          }
          os << Base->getType().getAsString(policy);

          if (Base->isPackExpansion())
            os << "...";
        }
      }

      // Print the class definition
      // FIXME: Doesn't print access specifiers, e.g., "public:"
      if (policy.TerseOutput) {
        os << " {}";
      } else {
        os << " {\n";
        VisitDeclContext(D);
        indent() << '}';
      }
    }
  }

  void VisitLinkageSpecDecl(clang::LinkageSpecDecl *D) {
    const char *l;
    if (D->getLanguage() == clang::LinkageSpecDecl::lang_c)
      l = "C";
    else {
      assert(D->getLanguage() == clang::LinkageSpecDecl::lang_cxx &&
             "unknown language in linkage specification");
      l = "C++";
    }

    os << "extern \"" << l << "\" ";
    if (D->hasBraces()) {
      os << "{\n";
      VisitDeclContext(D);
      indent() << '}';
    } else
      Visit(*D->decls_begin());
  }

  void printTemplateParameters(const clang::TemplateParameterList *Params) {
    assert(Params);

    os << "template <";

    for (unsigned i = 0, e = Params->size(); i != e; ++i) {
      if (i != 0)
        os << ", ";

      const auto* Param = Params->getParam(i);
      if (auto TTP = clang::dyn_cast<clang::TemplateTypeParmDecl>(Param)) {

        if (TTP->wasDeclaredWithTypename())
          os << "typename ";
        else
          os << "class ";

        if (TTP->isParameterPack())
          os << "...";

        os << *TTP;

        if (TTP->hasDefaultArgument()) {
          os << " = ";
          os << TTP->getDefaultArgument().getAsString(policy);
        };
      } else if (auto NTTP = clang::dyn_cast<clang::NonTypeTemplateParmDecl>(Param)) {
        llvm::StringRef Name;
        if (clang::IdentifierInfo *II = NTTP->getIdentifier())
          Name = II->getName();
        printDeclType(NTTP->getType(), Name, NTTP->isParameterPack());

        if (NTTP->hasDefaultArgument()) {
          os << " = ";
          sv.Visit(NTTP->getDefaultArgument());
        }
      } else if (auto TTPD = clang::dyn_cast<clang::TemplateTemplateParmDecl>(Param)) {
        VisitTemplateDecl(TTPD);
        // FIXME: print the default argument, if present.
      }
    }

    os << "> ";
  }

  void printTemplateArguments(const clang::TemplateArgumentList &Args,
                              const clang::TemplateParameterList *Params = nullptr) {
    os << '<';
    for (size_t I = 0, E = Args.size(); I < E; ++I) {
      const clang::TemplateArgument &A = Args[I];
      if (I)
        os << ", ";
      if (Params) {
        if (A.getKind() == clang::TemplateArgument::Type)
          if (auto T = A.getAsType()->getAs<clang::TemplateTypeParmType>()) {
            auto P = clang::cast<clang::TemplateTypeParmDecl>(Params->getParam(T->getIndex()));
            os << *P;
            continue;
          }
        if (A.getKind() == clang::TemplateArgument::Template) {
          if (auto T = A.getAsTemplate().getAsTemplateDecl())
            if (auto TD = clang::dyn_cast<clang::TemplateTemplateParmDecl>(T)) {
              auto P = clang::cast<clang::TemplateTemplateParmDecl>(
                                                Params->getParam(TD->getIndex()));
              os << *P;
              continue;
            }
        }
        if (A.getKind() == clang::TemplateArgument::Expression) {
          if (auto E = clang::dyn_cast<clang::DeclRefExpr>(A.getAsExpr()))
            if (auto N = clang::dyn_cast<clang::NonTypeTemplateParmDecl>(E->getDecl())) {
              auto P = clang::cast<clang::NonTypeTemplateParmDecl>(
                                                 Params->getParam(N->getIndex()));
              os << *P;
              continue;
            }
        }
      }
      A.print(policy, os);
    }
    os << '>';
  }

  void VisitTemplateDecl(const clang::TemplateDecl *D) {
    printTemplateParameters(D->getTemplateParameters());

    if(auto TTP = clang::dyn_cast<clang::TemplateTemplateParmDecl>(D)){
      os << "class ";
      if (TTP->isParameterPack())
        os << "...";
      os << D->getName();
    } else {
      Visit(D->getTemplatedDecl());
    }
  }

  void VisitFunctionTemplateDecl(clang::FunctionTemplateDecl *D) {
    prettyPrintPragmas(D->getTemplatedDecl());
    // Print any leading template parameter lists.
    if(auto FD = D->getTemplatedDecl()) {
      for (unsigned I = 0, NumTemplateParams = FD->getNumTemplateParameterLists();
           I < NumTemplateParams; ++I)
        printTemplateParameters(FD->getTemplateParameterList(I));
    }
    VisitRedeclarableTemplateDecl(D);

    // Never print "instantiations" for deduction guides (they don't really
    // have them).
    if (PrintInstantiation) {
      clang::FunctionDecl *PrevDecl = D->getTemplatedDecl();
      const clang::FunctionDecl *Def;
      if (PrevDecl->isDefined(Def) && Def != PrevDecl)
        return;
      for (auto *I : D->specializations())
        if (I->getTemplateSpecializationKind() == clang::TSK_ImplicitInstantiation) {
          if (!PrevDecl->isThisDeclarationADefinition())
            os << ";\n";
          indent();
          prettyPrintPragmas(I);
          Visit(I);
        }
    }
  }

  void VisitClassTemplateDecl(clang::ClassTemplateDecl *D) {
    VisitRedeclarableTemplateDecl(D);

    if (PrintInstantiation)
      for (auto *I : D->specializations())
        if (I->getSpecializationKind() == clang::TSK_ImplicitInstantiation) {
          if (D->isThisDeclarationADefinition())
            os << ';';
          os << '\n';
          Visit(I);
        }
  }

  void VisitClassTemplateSpecializationDecl(clang::ClassTemplateSpecializationDecl *D) {
    os << "template<> ";
    VisitCXXRecordDecl(D);
  }

  void VisitClassTemplatePartialSpecializationDecl(
                                      clang::ClassTemplatePartialSpecializationDecl *D) {
    printTemplateParameters(D->getTemplateParameters());
    VisitCXXRecordDecl(D);
  }

  void VisitUsingDecl(clang::UsingDecl *D) {
    if (!D->isAccessDeclaration())
      os << "using ";
    if (D->hasTypename())
      os << "typename ";
    D->getQualifier()->print(os, policy);

    // Use the correct record name when the using declaration is used for
    // inheriting constructors.
    for (const auto *Shadow : D->shadows()) {
      if(const auto *ConstructorShadow =
              clang::dyn_cast<clang::ConstructorUsingShadowDecl>(Shadow)) {
        assert(Shadow->getDeclContext() == ConstructorShadow->getDeclContext());
        os << *ConstructorShadow->getNominatedBaseClass();
        return;
      }
    }
    os << *D;
  }

  void
  VisitUnresolvedUsingTypenameDecl(clang::UnresolvedUsingTypenameDecl *D) {
    os << "using typename ";
    D->getQualifier()->print(os, policy);
    os << D->getDeclName();
  }

  void VisitUnresolvedUsingValueDecl(clang::UnresolvedUsingValueDecl *D) {
    if (!D->isAccessDeclaration())
      os << "using ";
    D->getQualifier()->print(os, policy);
    os << D->getDeclName();
  }

  void VisitUsingShadowDecl(clang::UsingShadowDecl*) {}
};

namespace registrar{

class ast_consumer : public clang::ASTConsumer{
  std::unique_ptr<decl_visitor> visit;
  static clang::PrintingPolicy ppolicy(clang::PrintingPolicy pp){
    pp.Bool = true;
    return pp;
  }
 public:
  explicit ast_consumer(clang::CompilerInstance& ci) : visit{new decl_visitor{llvm::outs(), ppolicy(ci.getASTContext().getPrintingPolicy())}}{}
  virtual void HandleTranslationUnit(clang::ASTContext& context)override{
    visit->Visit(context.getTranslationUnitDecl());
  }
};

struct ast_frontend_action : clang::SyntaxOnlyAction{
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& ci, clang::StringRef)override{
    return llvm::make_unique<ast_consumer>(ci);
  }
};

}

}

int main(int argc, const char** argv){
  llvm::cl::OptionCategory tool_category("ultima options");
  llvm::cl::extrahelp common_help(clang::tooling::CommonOptionsParser::HelpMessage);
  std::vector<const char*> params;
  params.reserve(argc+1);
  std::copy(argv, argv+argc, std::back_inserter(params));
  params.emplace_back("-D__ULTIMA=1");
  params.emplace_back("-xc++");
  params.emplace_back("-std=c++14");
  params.emplace_back("-w");
  params.emplace_back("-Wno-narrowing");
  params.emplace_back("-includecl_stub.hpp");
  params.emplace_back("-includecuda_stub.hpp");
  clang::tooling::CommonOptionsParser options_parser(argc = static_cast<int>(params.size()), params.data(), tool_category);
  clang::tooling::ClangTool tool(options_parser.getCompilations(), options_parser.getSourcePathList());
  return tool.run(clang::tooling::newFrontendActionFactory<ultima::registrar::ast_frontend_action>().get());
}
