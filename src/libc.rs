//! Signatures of common C library functions, for calls into imports. Without it a
//! call to an import passes the argument registers set up in the call's own block
//! (`abi::guess_args`), which misses arguments the caller passes through unchanged
//! (`free(p)` at the start of a function that took `p`).
//!
//! Only the integer/pointer shape matters here: how many argument registers, and
//! whether rax holds a result.
use crate::abi::Sig;

/// (name, register arguments, returns a value, variadic)
const TABLE: &[(&str, u8, bool, bool)] = &[
    // memory
    ("malloc", 1, true, false), ("calloc", 2, true, false), ("realloc", 2, true, false),
    ("reallocarray", 3, true, false), ("free", 1, false, false), ("aligned_alloc", 2, true, false),
    ("posix_memalign", 3, true, false), ("memcpy", 3, true, false), ("memmove", 3, true, false),
    ("memset", 3, true, false), ("memcmp", 3, true, false), ("memchr", 3, true, false),
    ("memrchr", 3, true, false), ("bcmp", 3, true, false), ("explicit_bzero", 2, false, false),
    ("__memcpy_chk", 4, true, false), ("__memmove_chk", 4, true, false), ("__memset_chk", 4, true, false),
    ("mmap", 6, true, false), ("munmap", 2, true, false), ("mprotect", 3, true, false),
    // strings
    ("strlen", 1, true, false), ("strnlen", 2, true, false), ("strcmp", 2, true, false),
    ("strncmp", 3, true, false), ("strcasecmp", 2, true, false), ("strncasecmp", 3, true, false),
    ("strcpy", 2, true, false), ("strncpy", 3, true, false), ("stpcpy", 2, true, false),
    ("strcat", 2, true, false), ("strncat", 3, true, false), ("strchr", 2, true, false),
    ("strrchr", 2, true, false), ("strstr", 2, true, false), ("strdup", 1, true, false),
    ("strndup", 2, true, false), ("strtok", 2, true, false), ("strtok_r", 3, true, false),
    ("strspn", 2, true, false), ("strcspn", 2, true, false), ("strpbrk", 2, true, false),
    ("strerror", 1, true, false), ("strtol", 3, true, false), ("strtoul", 3, true, false),
    ("strtoll", 3, true, false), ("strtoull", 3, true, false), ("atoi", 1, true, false),
    ("atol", 1, true, false), ("atoll", 1, true, false), ("toupper", 1, true, false),
    ("tolower", 1, true, false), ("isalpha", 1, true, false), ("isdigit", 1, true, false),
    ("isspace", 1, true, false), ("isalnum", 1, true, false), ("isupper", 1, true, false),
    ("islower", 1, true, false), ("isprint", 1, true, false), ("isxdigit", 1, true, false),
    ("__ctype_b_loc", 0, true, false), ("__ctype_toupper_loc", 0, true, false),
    ("__ctype_tolower_loc", 0, true, false), ("__strcpy_chk", 3, true, false),
    ("__strcat_chk", 3, true, false), ("__strncpy_chk", 4, true, false),
    // stdio
    ("printf", 1, true, true), ("fprintf", 2, true, true), ("sprintf", 2, true, true),
    ("snprintf", 3, true, true), ("dprintf", 2, true, true), ("asprintf", 2, true, true),
    ("__printf_chk", 2, true, true), ("__fprintf_chk", 3, true, true), ("__sprintf_chk", 4, true, true),
    ("__snprintf_chk", 5, true, true), ("scanf", 1, true, true), ("sscanf", 2, true, true),
    ("fscanf", 2, true, true), ("__isoc99_sscanf", 2, true, true), ("__isoc99_scanf", 1, true, true),
    ("__isoc99_fscanf", 2, true, true), ("__isoc23_sscanf", 2, true, true),
    ("__isoc23_strtol", 3, true, false), ("__isoc23_strtoul", 3, true, false),
    ("vprintf", 2, true, false), ("vfprintf", 3, true, false), ("vsnprintf", 4, true, false),
    ("vsprintf", 3, true, false), ("puts", 1, true, false), ("fputs", 2, true, false),
    ("putchar", 1, true, false), ("fputc", 2, true, false), ("putc", 2, true, false),
    ("getchar", 0, true, false), ("fgetc", 1, true, false), ("getc", 1, true, false),
    ("fgets", 3, true, false), ("fopen", 2, true, false), ("fdopen", 2, true, false),
    ("fclose", 1, true, false), ("fread", 4, true, false), ("fwrite", 4, true, false),
    ("fflush", 1, true, false), ("fseek", 3, true, false), ("ftell", 1, true, false),
    ("rewind", 1, false, false), ("feof", 1, true, false), ("ferror", 1, true, false),
    ("perror", 1, false, false), ("setvbuf", 4, true, false), ("getline", 3, true, false),
    ("ungetc", 2, true, false), ("fileno", 1, true, false),
    // process, system
    ("exit", 1, false, false), ("_exit", 1, false, false), ("abort", 0, false, false),
    ("atexit", 1, true, false), ("__cxa_atexit", 3, true, false), ("__cxa_finalize", 1, false, false),
    ("__stack_chk_fail", 0, false, false), ("__assert_fail", 4, false, false),
    ("getenv", 1, true, false), ("setenv", 3, true, false), ("system", 1, true, false),
    ("open", 2, true, true), ("openat", 3, true, true), ("close", 1, true, false),
    ("read", 3, true, false), ("write", 3, true, false), ("lseek", 3, true, false),
    ("pread", 4, true, false), ("pwrite", 4, true, false), ("fstat", 2, true, false),
    ("stat", 2, true, false), ("lstat", 2, true, false), ("unlink", 1, true, false),
    ("ioctl", 2, true, true), ("fcntl", 2, true, true), ("getpid", 0, true, false),
    ("fork", 0, true, false), ("execve", 3, true, false), ("execvp", 2, true, false),
    ("waitpid", 3, true, false), ("kill", 2, true, false), ("signal", 2, true, false),
    ("sigaction", 3, true, false), ("raise", 1, true, false), ("syscall", 1, true, true),
    ("sleep", 1, true, false), ("usleep", 1, true, false), ("nanosleep", 2, true, false),
    ("time", 1, true, false), ("clock_gettime", 2, true, false), ("gettimeofday", 2, true, false),
    ("clock", 0, true, false), ("rand", 0, true, false), ("srand", 1, false, false),
    ("random", 0, true, false), ("srandom", 1, false, false), ("__errno_location", 0, true, false),
    ("qsort", 4, false, false), ("bsearch", 5, true, false), ("abs", 1, true, false),
    ("labs", 1, true, false), ("dlopen", 2, true, false), ("dlsym", 2, true, false),
    ("dlclose", 1, true, false), ("dlerror", 0, true, false), ("getauxval", 1, true, false),
    ("sysconf", 1, true, false), ("isatty", 1, true, false), ("pipe", 1, true, false),
    ("dup", 1, true, false), ("dup2", 2, true, false), ("chdir", 1, true, false),
    ("getcwd", 2, true, false), ("mkdir", 2, true, false), ("rmdir", 1, true, false),
    ("opendir", 1, true, false), ("readdir", 1, true, false), ("closedir", 1, true, false),
    ("realpath", 2, true, false), ("poll", 3, true, false), ("socket", 3, true, false),
    ("connect", 3, true, false), ("bind", 3, true, false), ("listen", 2, true, false),
    ("accept", 3, true, false), ("send", 4, true, false), ("recv", 4, true, false),
    ("setlocale", 2, true, false), ("__libc_start_main", 6, true, false),
    // threads
    ("pthread_create", 4, true, false), ("pthread_join", 2, true, false),
    ("pthread_self", 0, true, false), ("pthread_mutex_lock", 1, true, false),
    ("pthread_mutex_unlock", 1, true, false), ("pthread_mutex_trylock", 1, true, false),
    ("pthread_mutex_init", 2, true, false), ("pthread_mutex_destroy", 1, true, false),
    ("pthread_cond_wait", 2, true, false), ("pthread_cond_signal", 1, true, false),
    ("pthread_cond_broadcast", 1, true, false), ("pthread_key_create", 2, true, false),
    ("pthread_getspecific", 1, true, false), ("pthread_setspecific", 2, true, false),
    ("pthread_once", 2, true, false), ("pthread_attr_init", 1, true, false),
    ("pthread_attr_destroy", 1, true, false), ("pthread_getattr_np", 2, true, false),
    ("pthread_attr_getstack", 3, true, false), ("pthread_sigmask", 3, true, false),
    ("__tls_get_addr", 1, true, false), ("sched_yield", 0, true, false),
    // C++ runtime and unwinding
    ("_Znwm", 1, true, false), ("_Znam", 1, true, false), ("_ZdlPv", 1, false, false),
    ("_ZdaPv", 1, false, false), ("_ZdlPvm", 2, false, false), ("__cxa_allocate_exception", 1, true, false),
    ("__cxa_throw", 3, false, false), ("__cxa_begin_catch", 1, true, false),
    ("__cxa_end_catch", 0, false, false), ("__cxa_rethrow", 0, false, false),
    ("__cxa_guard_acquire", 1, true, false), ("__cxa_guard_release", 1, false, false),
    ("__cxa_pure_virtual", 0, false, false), ("__cxa_thread_atexit_impl", 3, true, false),
    ("_Unwind_Resume", 1, false, false), ("_Unwind_RaiseException", 1, true, false),
    ("_Unwind_GetIP", 1, true, false), ("_Unwind_GetCFA", 1, true, false),
    ("_Unwind_GetLanguageSpecificData", 1, true, false), ("_Unwind_GetRegionStart", 1, true, false),
    ("_Unwind_SetGR", 3, false, false), ("_Unwind_SetIP", 2, false, false),
    ("_Unwind_Backtrace", 2, true, false), ("_Unwind_DeleteException", 1, false, false),
];

/// The signature of a C library function, if the table knows it.
pub fn lookup(name: &str) -> Option<Sig> {
    // versioned names from some symbol tables: memcpy@GLIBC_2.14
    let name = name.split('@').next().unwrap_or(name);
    TABLE
        .iter()
        .find(|e| e.0 == name)
        .map(|&(_, args, ret, variadic)| Sig { args, ret, variadic, ..Default::default() })
}

/// Safe mode's summary of an allocator or deallocator, by name: `malloc` and
/// friends return a new allocation (a `Box` when the caller uses it like one),
/// `free` and friends consume their first argument. Other functions have none.
pub fn summary(name: &str) -> Option<crate::borrow::Callee> {
    use crate::borrow::{Callee, Pass};
    let name = name.split('@').next().unwrap_or(name);
    let alloc = |size: (u8, Option<u8>), args: usize| Some(Callee { args: vec![Pass::Ignore; args], alloc: Some(size), ret_from: 0, raw: false });
    match name {
        "malloc" | "_Znwm" | "_Znam" => alloc((0, None), 1),
        "calloc" => alloc((0, Some(1)), 2),
        "__rust_alloc" | "__rust_alloc_zeroed" => alloc((0, None), 2),
        "free" | "_ZdlPv" | "_ZdaPv" | "_ZdlPvm" | "_ZdaPvm" | "__rust_dealloc" => {
            let mut args = vec![Pass::Ignore; 3];
            args[0] = Pass::Free;
            Some(Callee { args, alloc: None, ret_from: 0, raw: false })
        }
        "memcpy" | "memmove" => Some(Callee {
            args: vec![Pass::Access { write: true }, Pass::Access { write: false }, Pass::Ignore],
            alloc: None,
            ret_from: 1,
            raw: false,
        }),
        "memset" => Some(Callee { args: vec![Pass::Access { write: true }, Pass::Ignore, Pass::Ignore], alloc: None, ret_from: 1, raw: false }),
        _ => None,
    }
}

/// C library functions safe mode writes as slice operations when their pointer
/// arguments have safe roots.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Builtin {
    /// `memcpy(dst, src, n)` / `memmove`: `copy_from_slice` or `copy_within`.
    Copy,
    /// `memset(dst, c, n)`: `fill`.
    Fill,
}

pub fn builtin(name: &str) -> Option<Builtin> {
    match name.split('@').next().unwrap_or(name) {
        "memcpy" | "memmove" => Some(Builtin::Copy),
        "memset" => Some(Builtin::Fill),
        _ => None,
    }
}
