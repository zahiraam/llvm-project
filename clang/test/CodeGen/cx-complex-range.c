// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=limited -o - | FileCheck %s --check-prefix=LMTD

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=smith -o - | FileCheck %s --check-prefix=FRTRN

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=extend -o - | FileCheck %s --check-prefix=EXTND

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=full -o - | FileCheck %s --check-prefix=FULL

// Fast math
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=limited -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=LMTD

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=full -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=smith -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=FRTRN

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=extend -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=EXTND

// LABEL: define {{.*}} @div(
// FULL: call {{.*}} @__divsc3
//
// LMTD: fmul{{.*}}float
// LMTD-NEXT: fmul{{.*}}float
// LMTD-NEXT: fadd{{.*}}float
// LMTD-NEXT: fmul{{.*}}float
// LMTD-NEXT: fmul{{.*}}float
// LMTD-NEXT: fadd{{.*}}float
// LMTD-NEXT: fmul{{.*}}float
// LMTD-NEXT: fmul{{.*}}float
// LMTD-NEXT: fsub{{.*}}float
// LMTD-NEXT: fdiv{{.*}}float
// LMTD-NEXT: fdiv{{.*}}float
//
// FRTRN: call{{.*}}float @llvm.fabs.f32(float {{.*}})
// FRTRN-NEXT: call{{.*}}float @llvm.fabs.f32(float {{.*}})
// FRTRN-NEXT: fcmp{{.*}}ugt float {{.*}}, {{.*}}
// FRTRN-NEXT:   br i1 {{.*}}, label
// FRTRN:  abs_rhsr_greater_or_equal_abs_rhsi:
// FRTRN-NEXT: fdiv{{.*}}float
// FRTRN-NEXT: fmul{{.*}}float
// FRTRN-NEXT: fadd{{.*}}float
// FRTRN-NEXT: fmul{{.*}}float
// FRTRN-NEXT: fadd{{.*}}float
// FRTRN-NEXT: fdiv{{.*}}float
// FRTRN-NEXT: fmul{{.*}}float
// FRTRN-NEXT: fsub{{.*}}float
// FRTRN-NEXT: fdiv{{.*}}float
// FRTRN-NEXT: br label
// FRTRN: abs_rhsr_less_than_abs_rhsi:
// FRTRN-NEXT: fdiv{{.*}}float
// FRTRN-NEXT: fmul{{.*}}float
// FRTRN-NEXT: fadd{{.*}}float
// FRTRN-NEXT: fmul{{.*}}float
// FRTRN-NEXT: fadd{{.*}}float
// FRTRN-NEXT: fdiv{{.*}}float
// FRTRN-NEXT: fmul{{.*}}float
// FRTRN-NEXT: fsub{{.*}}float
// FRTRN-NEXT: fdiv{{.*}}float
//
// EXTND: load float, ptr {{.*}}
// EXTND: fpext float {{.*}} to double
// EXTND-NEXT: fpext float {{.*}} to double
// EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
// EXTND-NEXT: load float, ptr {{.*}}
// EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
// EXTND-NEXT: load float, ptr {{.*}}
// EXTND-NEXT: fpext float {{.*}} to double
// EXTND-NEXT: fpext float {{.*}} to double
// EXTND-NEXT: fmul{{.*}}double
// EXTND-NEXT: fmul{{.*}}double
// EXTND-NEXT: fadd{{.*}}double
// EXTND-NEXT: fmul{{.*}}double
// EXTND-NEXT: fmul{{.*}}double
// EXTND-NEXT: fadd{{.*}}double
// EXTND-NEXT: fmul{{.*}}double
// EXTND-NEXT: fmul{{.*}}double
// EXTND-NEXT: fsub{{.*}}double
// EXTND-NEXT: fdiv{{.*}}double
// EXTND-NEXT: fdiv{{.*}}double
// EXTND-NEXT: fptrunc double {{.*}} to float
// EXTND-NEXT: fptrunc double {{.*}} to float
//
_Complex float div(_Complex float a, _Complex float b) {
  return a / b;
}

// LABEL: define {{.*}} @mul(
// FULL: call {{.*}} @__mulsc3
//
// LMTD: alloca { float, float }
// LMTD-NEXT: alloca { float, float }
// LMTD-NEXT: alloca { float, float }
// LMTD: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
// LMTD-NEXT: load float, ptr {{.*}}
// LMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
// LMTD-NEXT: load float, ptr {{.*}}
// LMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
// LMTD-NEXT: load float, ptr {{.*}}
// LMTD-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
// LMTD-NEXT: load float
// LMTD-NEXT: fmul{{.*}}float
// LMTD-NEXT: fmul{{.*}}float
// LMTD-NEXT: fmul{{.*}}float
// LMTD-NEXT: fmul{{.*}}float
// LMTD-NEXT: fsub{{.*}}float
// LMTD-NEXT: fadd{{.*}}float
//
// FRTRN: alloca { float, float }
// FRTRN-NEXT: alloca { float, float }
// FRTRN-NEXT: alloca { float, float }
// FRTRN: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
// FRTRN-NEXT: load float, ptr {{.*}}
// FRTRN-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
// FRTRN-NEXT: load float, ptr {{.*}}
// FRTRN-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
// FRTRN-NEXT: load float, ptr {{.*}}
// FRTRN-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
// FRTRN-NEXT: load float
// FRTRN-NEXT: fmul{{.*}}float
// FRTRN-NEXT: fmul{{.*}}float
// FRTRN-NEXT: fmul{{.*}}float
// FRTRN-NEXT: fmul{{.*}}float
// FRTRN-NEXT: fsub{{.*}}float
// FRTRN-NEXT: fadd{{.*}}float
//
// EXTND: alloca { float, float }
// EXTND-NEXT: alloca { float, float }
// EXTND-NEXT: alloca { float, float }
// EXTND: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
// EXTND-NEXT: load float, ptr
// EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
// EXTND-NEXT: load float, ptr {{.*}}
// EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
// EXTND-NEXT: load float, ptr {{.*}}
// EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
// EXTND-NEXT: load{{.*}}float
// EXTND-NEXT: fmul{{.*}}float
// EXTND-NEXT: fmul{{.*}}float
// EXTND-NEXT: fmul{{.*}}float
// EXTND-NEXT: fmul{{.*}}float
// EXTND-NEXT: fsub{{.*}}float
// EXTND-NEXT: fadd{{.*}}float
//
_Complex float mul(_Complex float a, _Complex float b) {
  return a * b;
}
