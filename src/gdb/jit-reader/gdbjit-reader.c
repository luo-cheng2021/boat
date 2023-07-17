#include <gdb/jit-reader.h>

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <memory.h>
#include <stdio.h>


GDB_DECLARE_GPL_COMPATIBLE_READER

// ref: https://github.com/erlang/otp/blob/master/erts/etc/unix/jit-reader.c

struct line_info {
	char name[128];
	uint64_t addr;
	int32_t line_num;
};

uint64_t g_start = 0;
uint64_t g_end = 0;
enum gdb_status read_debug_info(struct gdb_reader_funcs *self, struct gdb_symbol_callbacks *cb, void *memory, long memory_sz){
	// get begin and end of code segment
	GDB_CORE_ADDR begin = *(GDB_CORE_ADDR*)memory;
	memory += sizeof(GDB_CORE_ADDR);
	GDB_CORE_ADDR end = *(GDB_CORE_ADDR*)memory;
	memory += sizeof(GDB_CORE_ADDR);
	uint64_t line_no = *(uint64_t*)memory;
	memory += sizeof(uint64_t);
	g_start = begin;
	g_end = end;
	// get name of function, just a single one per file
	const char *name = (const char*)memory;
	int name_len = strlen(name) + 1;
	struct line_info *p = (struct line_info *)(memory + name_len);

	struct gdb_object *obj = cb->object_open(cb);

	struct gdb_symtab *symtab = cb->symtab_open(cb, obj, "gdbjit");
	struct gdb_block * b = cb->block_open(cb, symtab, NULL, begin, end, name);

	cb->symtab_close(cb, symtab);
	for (uint64_t i = 0; i < line_no; i++) {
		struct gdb_line_mapping line_mapping;
		line_mapping.pc = p[i].addr;
		line_mapping.line = p[i].line_num;
		// printf("%i %s %llx %d\n", (int32_t)i, p[i].name, p[i].addr, p[i].line_num);

		struct gdb_symtab *symtab = cb->symtab_open(cb, obj, p[i].name);
		// returned value has no use
		cb->block_open(cb, symtab, NULL, p[i].addr, end, name);
		cb->line_mapping_add(cb, symtab, 1, &line_mapping);
		cb->symtab_close(cb, symtab);
	}
	cb->object_close(cb, obj);
	return GDB_SUCCESS;
}

typedef enum {
    X64_RBP = 6,  /* Frame pointer iff native frames are enabled */
    X64_RSP = 7,  /* Stack pointer when using native stack */
    X64_R12 = 12, /* Stack pointer when using non-native stack */
    X64_R13 = 13, /* Current process */
    X64_RIP = 16
} X64Register;

static void regfree(struct gdb_reg_value *reg) {
    free(reg);
}

enum gdb_status unwind(struct gdb_reader_funcs *self, struct gdb_unwind_callbacks *cb){
	//TODO
    GDB_CORE_ADDR rbp, rsp, rip;
    struct range *range;

    rbp = *(GDB_CORE_ADDR*)cb->reg_get(cb, X64_RBP)->value;
    rsp = *(GDB_CORE_ADDR*)cb->reg_get(cb, X64_RSP)->value;
    rip = *(GDB_CORE_ADDR*)cb->reg_get(cb, X64_RIP)->value;
	// printf("org rip =  0x%lx org rsp  0x%lx, org rbp 0x%lx\n", rip, rsp, rbp);

	if (g_start && rip >= g_start && rip <= g_end) {
	    struct gdb_reg_value *prev_rbp, *prev_rsp, *prev_rip;
		prev_rbp = malloc(sizeof(struct gdb_reg_value) + sizeof(char*));
		prev_rsp = malloc(sizeof(struct gdb_reg_value) + sizeof(char*));
		prev_rip = malloc(sizeof(struct gdb_reg_value) + sizeof(char*));
		prev_rbp->free = &regfree;
		prev_rbp->defined = 1;
		prev_rbp->size = sizeof(char*);
		prev_rsp->free = &regfree;
		prev_rsp->defined = 1;
		prev_rsp->size = sizeof(char*);
		prev_rip->free = &regfree;
		prev_rip->defined = 1;
		prev_rip->size = sizeof(char*);
		cb->target_read(rsp + 0 * sizeof(char*), &prev_rip->value,
						sizeof(char*));
		// cb->target_read(rbp + 0 * sizeof(char*), &prev_rbp->value,
		// 				sizeof(char*));
		*(GDB_CORE_ADDR*)prev_rsp->value = rsp + sizeof(char*);
		*(GDB_CORE_ADDR*)prev_rbp->value = rbp;

		cb->reg_set(cb, X64_RIP, prev_rip);
		cb->reg_set(cb, X64_RSP, prev_rsp);
		cb->reg_set(cb, X64_RBP, prev_rbp);
		// printf("rip = 0x%lx prev rip 0x%lx, prev rsp 0x%lx, prev rbp 0x%lx\n", rip, *(GDB_CORE_ADDR*)prev_rip->value, *(GDB_CORE_ADDR*)prev_rsp->value, *(GDB_CORE_ADDR*)prev_rbp->value);
		// printf("org rip =  0x%lx org rsp  0x%lx, org rbp 0x%lx\n", rip, rsp, rbp);
		return GDB_SUCCESS;
	}

	return GDB_FAIL;
}

struct gdb_frame_id get_frame_id(struct gdb_reader_funcs *self, struct gdb_unwind_callbacks *cb){
    struct gdb_frame_id frame = {0, 0};
    GDB_CORE_ADDR rbp, rsp, rip;

    rbp = *(GDB_CORE_ADDR*)cb->reg_get(cb, X64_RBP)->value;
    rsp = *(GDB_CORE_ADDR*)cb->reg_get(cb, X64_RSP)->value;
    rip = *(GDB_CORE_ADDR*)cb->reg_get(cb, X64_RIP)->value;

    // printf("FRAME: rip: 0x%lx rsp: 0x%lx rbp: 0x%lx \r\n", rip, rsp, rbp);

	if (g_start && rip >= g_start && rip <= g_end) {
        frame.code_address = rip;

		frame.stack_address = rsp;// + sizeof(char*);
            // GDB_CORE_ADDR prev_rip;

            // for (rsp += sizeof(char*); ; rsp += sizeof(char*)) {
            //     cb->target_read(rsp, &prev_rip, sizeof(char*));

            //     printf("rsp: 0x%lx rip: 0x%lx\r\n", rsp, prev_rip);

            //     /* Check if it is a cp */
            //     if ((prev_rip & 0x3) == 0) {
            //         break;
            //     }
            // }

            // frame.stack_address = rsp;
    }

    // printf("FRAME: code_address: 0x%lx stack_address: 0x%lx\r\n",
    //     frame.code_address, frame.stack_address);

    return frame;
}

void destroy(struct gdb_reader_funcs *self){
	free(self);
}


struct gdb_reader_funcs *gdb_init_reader(void){
	struct gdb_reader_funcs *funcs = malloc(sizeof(struct gdb_reader_funcs));
	funcs->reader_version = GDB_READER_INTERFACE_VERSION;
	funcs->priv_data = NULL;

	funcs->read = read_debug_info;
	funcs->unwind = unwind;
	funcs->get_frame_id = get_frame_id;
	funcs->destroy = destroy;

	return funcs;
}
