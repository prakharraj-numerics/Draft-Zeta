#define main zeta53_local_center_embedded_main
#include "zeta53_local_center.cpp"
#undef main

int main(){
    SourceTables src;
    LocalTables lt(src);
    std::puts("#pragma once");
    std::puts("#include <cstddef>");
    std::puts("namespace zeta53_local9_table {");
    std::printf("inline constexpr int NBINS = %d;\n", NBINS);
    std::printf("inline constexpr int DEGREE = 9;\n");
    std::printf("inline constexpr double BINW = %a;\n", BINW);
    std::printf("inline constexpr double X_CENTER0 = %a;\n", A0-2.0);
    std::puts("alignas(64) inline constexpr double C[DEGREE+1][NBINS] = {");
    for(int k=0;k<=9;k++){
        std::puts("  {");
        for(int b=0;b<NBINS;b++){
            std::printf("    %a%s", lt.hi[k][b], (b+1==NBINS)?"\n":",\n");
        }
        std::printf("  }%s\n", (k==9)?"":",");
    }
    std::puts("};");
    std::puts("} // namespace zeta53_local9_table");
    flint_cleanup();
    return 0;
}
