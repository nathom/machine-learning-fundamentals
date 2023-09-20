#include "matrix.h"

int
main()
{
    matrix m1;
    INIT_ZEROS(m1, 2, 2);
    matrix m2;
    INIT_ID(m2, 2);
    matrix_print(m1);
    matrix_print(m2);

    matrix_ip_add(m1, m2);
    matrix_print(m1);
}
