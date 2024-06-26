; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; Check that the constraints on the parameter derived from the
; range metadata (see bottom of the file) are present:
;
; CHECK: Context:
; CHECK:   [tmp] -> {  : 0 <= tmp <= 255 }
;
;    void jd(int *A, int *p /* in [0,256) */) {
;      for (int i = 0; i < 1024; i++)
;        A[i + *p] = i;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(ptr %A, ptr %p) {
entry:
  %tmp = load i32, ptr %p, align 4, !range !0
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add i32 %i.0, %tmp
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %idxprom
  store i32 %i.0, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

!0 =  !{ i32 0, i32 256 }
