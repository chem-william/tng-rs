#include "tng/tng_io.h"

#ifdef USE_STD_INTTYPES_H
#    include <inttypes.h>
#endif

#include <stdint.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define NATOMS 100000
#define NFRAMES 20
#define CHUNKY 10
#define SCALE 0.5
#define PRECISION 1e-8
#define VELINTMUL 100000
#define TIME_PER_FRAME 2e-15
#define DEFAULT_REPS 10

static const int32_t INTMIN[3] = {0, 0, 0};
static const int32_t INTMAX[3] = {805306368, 805306368, 805306368};
static const int32_t INTSINTABLE[128] = {
    0, 3215, 6423, 9615, 12785, 15923, 19023, 22078, 25079, 28019, 30892, 33691, 36409, 39039,
    41574, 44010, 46340, 48558, 50659, 52638, 54490, 56211, 57796, 59242, 60546, 61704, 62713,
    63570, 64275, 64825, 65219, 65456, 65535, 65456, 65219, 64825, 64275, 63570, 62713, 61704,
    60546, 59242, 57796, 56211, 54490, 52638, 50659, 48558, 46340, 44010, 41574, 39039, 36409,
    33691, 30892, 28019, 25079, 22078, 19023, 15923, 12785, 9615, 6423, 3215, 0, -3215, -6423,
    -9615, -12785, -15923, -19023, -22078, -25079, -28019, -30892, -33691, -36409, -39039,
    -41574, -44010, -46340, -48558, -50659, -52638, -54490, -56211, -57796, -59242, -60546,
    -61704, -62713, -63570, -64275, -64825, -65219, -65456, -65535, -65456, -65219, -64825,
    -64275, -63570, -62713, -61704, -60546, -59242, -57796, -56211, -54490, -52638, -50659,
    -48558, -46340, -44010, -41574, -39039, -36409, -33691, -30892, -28019, -25079, -22078,
    -19023, -15923, -12785, -9615, -6423, -3215,
};

static void fail(const char* message)
{
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static size_t parse_reps(int argc, char** argv)
{
    char*              end = 0;
    unsigned long long parsed;

    if (argc < 2)
    {
        return DEFAULT_REPS;
    }

    errno  = 0;
    parsed = strtoull(argv[1], &end, 10);
    if (errno != 0 || end == argv[1] || (end && *end != '\0'))
    {
        return DEFAULT_REPS;
    }

    if (parsed > (unsigned long long)SIZE_MAX)
    {
        return DEFAULT_REPS;
    }

    return (size_t)parsed;
}

static int32_t intsin(int32_t i)
{
    if (i < 0)
    {
        return 0;
    }
    return INTSINTABLE[i % 128];
}

static int32_t intcos(int32_t i)
{
    return intsin((i < 0 ? 0 : i) + 32);
}

static void keepinbox(int32_t val[3])
{
    for (int dim = 0; dim < 3; dim++)
    {
        int32_t range = INTMAX[dim] - INTMIN[dim] + 1;
        while (val[dim] > INTMAX[dim])
        {
            val[dim] -= range;
        }
        while (val[dim] < INTMIN[dim])
        {
            val[dim] += range;
        }
    }
}

static void molecule(int32_t* target,
                     const int32_t base[3],
                     size_t        length,
                     int32_t       scale,
                     const int32_t direction[3],
                     int           flip,
                     int32_t       iframe)
{
    for (size_t i = 0; i < length; i++)
    {
        size_t ifl = i;

        if (flip && length > 1)
        {
            if (i == 0)
            {
                ifl = 1;
            }
            else if (i == 1)
            {
                ifl = 0;
            }
        }

        target[ifl * 3] = base[0] + (intsin(((int32_t)i + iframe) * direction[0]) * scale) / 256;
        target[ifl * 3 + 1] = base[1] + (intcos(((int32_t)i + iframe) * direction[1]) * scale) / 256;
        target[ifl * 3 + 2] = base[2] + (intcos(((int32_t)i + iframe) * direction[2]) * scale) / 256;

        int32_t atom[3] = {target[ifl * 3], target[ifl * 3 + 1], target[ifl * 3 + 2]};
        keepinbox(atom);
        target[ifl * 3] = atom[0];
        target[ifl * 3 + 1] = atom[1];
        target[ifl * 3 + 2] = atom[2];
    }
}

static void genibox(int32_t* intbox, int32_t iframe)
{
    size_t molecule_length = 1;
    int32_t molpos[3]      = {intsin(iframe) / 32, 1 + intcos(iframe) / 32, 2 + intsin(iframe) / 16};
    int32_t direction[3]   = {1, 1, 1};
    int32_t scale          = 1;
    int     flip           = 0;
    size_t  i              = 0;

    keepinbox(molpos);

    while (i < NATOMS)
    {
        size_t this_mol_length = molecule_length;

        if (this_mol_length > NATOMS - i)
        {
            this_mol_length = NATOMS - i;
        }

        if (i % 10 == 0)
        {
            intbox[i * 3] = molpos[0];
            intbox[i * 3 + 1] = molpos[1];
            intbox[i * 3 + 2] = molpos[2];
            for (size_t j = 1; j < this_mol_length; j++)
            {
                intbox[(i + j) * 3] = intbox[(i + j - 1) * 3] + (INTMAX[0] - INTMIN[0] + 1) / 5;
                intbox[(i + j) * 3 + 1] =
                    intbox[(i + j - 1) * 3 + 1] + (INTMAX[1] - INTMIN[1] + 1) / 5;
                intbox[(i + j) * 3 + 2] =
                    intbox[(i + j - 1) * 3 + 2] + (INTMAX[2] - INTMIN[2] + 1) / 5;

                int32_t atom[3] = {
                    intbox[(i + j) * 3],
                    intbox[(i + j) * 3 + 1],
                    intbox[(i + j) * 3 + 2],
                };
                keepinbox(atom);
                intbox[(i + j) * 3] = atom[0];
                intbox[(i + j) * 3 + 1] = atom[1];
                intbox[(i + j) * 3 + 2] = atom[2];
            }
        }
        else
        {
            molecule(&intbox[i * 3], molpos, this_mol_length, scale, direction, flip, iframe);
        }

        i += this_mol_length;

        molpos[0] += (intsin((int32_t)i * 3) < 0 ? -1 : 1) * (INTMAX[0] - INTMIN[0] + 1) / 20;
        molpos[1] += (intsin((int32_t)i * 5) < 0 ? -1 : 1) * (INTMAX[1] - INTMIN[1] + 1) / 20;
        molpos[2] += (intsin((int32_t)i * 7) < 0 ? -1 : 1) * (INTMAX[2] - INTMIN[2] + 1) / 20;
        keepinbox(molpos);

        direction[0] = ((direction[0] + 1) % 7) + 1;
        direction[1] = ((direction[1] + 1) % 3) + 1;
        direction[2] = ((direction[2] + 1) % 6) + 1;

        scale += 1;
        if (scale > 5)
        {
            scale = 1;
        }

        molecule_length += 1;
        if (molecule_length > 30)
        {
            molecule_length = 1;
        }
        if (i % 9 != 0)
        {
            flip = !flip;
        }
    }
}

static void genivelbox(int32_t* intvelbox, int32_t iframe)
{
    for (size_t i = 0; i < NATOMS; i++)
    {
        int32_t idx = (int32_t)i;
        intvelbox[i * 3] = (intsin((idx + iframe) * 3) / 10) * VELINTMUL + idx;
        intvelbox[i * 3 + 1] = 1 + (intcos((idx + iframe) * 5) / 10) * VELINTMUL + idx;
        intvelbox[i * 3 + 2] =
            2 + ((intsin((idx + iframe) * 7) + intcos((idx + iframe) * 9)) / 20) * VELINTMUL + idx;
    }
}

static void realbox(const int32_t* intbox, double* out)
{
    for (size_t i = 0; i < NATOMS; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            out[i * 3 + j] = (double)intbox[i * 3 + j] * PRECISION * SCALE;
        }
    }
}

static void bench_data(double* positions, double* velocities)
{
    const size_t frame_len = NATOMS * 3u;
    int32_t*     intbox = malloc(frame_len * sizeof(*intbox));
    int32_t*     intvelbox = malloc(frame_len * sizeof(*intvelbox));

    if (!intbox || !intvelbox)
    {
        free(intbox);
        free(intvelbox);
        fail("allocation failure");
    }

    for (int32_t frame = 0; frame < NFRAMES; frame++)
    {
        size_t start = (size_t)frame * frame_len;

        genibox(intbox, frame);
        genivelbox(intvelbox, frame);
        realbox(intbox, positions + start);
        realbox(intvelbox, velocities + start);
    }

    free(intbox);
    free(intvelbox);
}

static char* bench_output_path(void)
{
    const char* tmpdir = getenv("TMPDIR");
    const char* suffix = "tng_rs_flamegraph.tng";
    size_t      len;
    char*       path;

    if (!tmpdir || tmpdir[0] == '\0')
    {
        tmpdir = "/tmp";
    }

    len = strlen(tmpdir) + strlen(suffix) + 2;
    path = malloc(len);
    if (!path)
    {
        fail("allocation failure");
    }

    if (tmpdir[strlen(tmpdir) - 1] == '/')
    {
        snprintf(path, len, "%s%s", tmpdir, suffix);
    }
    else
    {
        snprintf(path, len, "%s/%s", tmpdir, suffix);
    }

    return path;
}

static uint64_t bench_write(const char* path, const double* positions, const double* velocities)
{
    const size_t    frame_len = NATOMS * 3u;
    tng_trajectory_t traj = 0;
    tng_molecule_t   molecule_handle = 0;
    tng_chain_t      chain_handle = 0;
    tng_residue_t    residue_handle = 0;
    tng_atom_t       atom_handle = 0;
    struct stat      st;
    const char*      error = 0;

    if (tng_trajectory_init(&traj) != TNG_SUCCESS)
    {
        error = "failed to initialize trajectory";
        goto cleanup;
    }
    if (tng_output_file_set(traj, path) != TNG_SUCCESS)
    {
        error = "failed to set output file";
        goto cleanup;
    }
    if (tng_num_frames_per_frame_set_set(traj, CHUNKY) != TNG_SUCCESS)
    {
        error = "failed to set frame-set size";
        goto cleanup;
    }
    if (tng_compression_precision_set(traj, 1.0 / PRECISION) != TNG_SUCCESS)
    {
        error = "failed to set compression precision";
        goto cleanup;
    }
    if (tng_time_per_frame_set(traj, TIME_PER_FRAME) != TNG_SUCCESS)
    {
        error = "failed to set time per frame";
        goto cleanup;
    }
    if (tng_molecule_add(traj, "particle", &molecule_handle) != TNG_SUCCESS)
    {
        error = "failed to add molecule";
        goto cleanup;
    }
    if (tng_molecule_chain_add(traj, molecule_handle, "A", &chain_handle) != TNG_SUCCESS)
    {
        error = "failed to add chain";
        goto cleanup;
    }
    if (tng_chain_residue_add(traj, chain_handle, "PAR", &residue_handle) != TNG_SUCCESS)
    {
        error = "failed to add residue";
        goto cleanup;
    }
    if (tng_residue_atom_add(traj, residue_handle, "P", "P", &atom_handle) != TNG_SUCCESS)
    {
        error = "failed to add atom";
        goto cleanup;
    }
    if (tng_molecule_cnt_set(traj, molecule_handle, NATOMS) != TNG_SUCCESS)
    {
        error = "failed to set molecule count";
        goto cleanup;
    }
    if (tng_file_headers_write(traj, TNG_SKIP_HASH) != TNG_SUCCESS)
    {
        error = "failed to write file headers";
        goto cleanup;
    }

    for (int64_t frame_start = 0; frame_start < NFRAMES; frame_start += CHUNKY)
    {
        int64_t frames_in_chunk = NFRAMES - frame_start;
        size_t  start = (size_t)frame_start * frame_len;
        size_t  chunk_values;
        size_t  chunk_bytes;
        void*   pos_bytes = 0;
        void*   vel_bytes = 0;

        if (frames_in_chunk > CHUNKY)
        {
            frames_in_chunk = CHUNKY;
        }

        chunk_values = (size_t)frames_in_chunk * frame_len;
        chunk_bytes = chunk_values * sizeof(double);
        pos_bytes = malloc(chunk_bytes);
        vel_bytes = malloc(chunk_bytes);
        if (!pos_bytes || !vel_bytes)
        {
            free(pos_bytes);
            free(vel_bytes);
            fail("allocation failure");
        }
        memcpy(pos_bytes, positions + start, chunk_bytes);
        memcpy(vel_bytes, velocities + start, chunk_bytes);

        if (tng_frame_set_with_time_new(traj, frame_start, frames_in_chunk, (double)frame_start * TIME_PER_FRAME)
            != TNG_SUCCESS)
        {
            free(pos_bytes);
            free(vel_bytes);
            error = "failed to create frame set";
            goto cleanup;
        }
        if (tng_particle_data_block_add(traj,
                                        TNG_TRAJ_POSITIONS,
                                        "POSITIONS",
                                        TNG_DOUBLE_DATA,
                                        TNG_TRAJECTORY_BLOCK,
                                        frames_in_chunk,
                                        3,
                                        1,
                                        0,
                                        NATOMS,
                                        TNG_TNG_COMPRESSION,
                                        pos_bytes)
            != TNG_SUCCESS)
        {
            free(pos_bytes);
            free(vel_bytes);
            error = "failed to add positions block";
            goto cleanup;
        }
        if (tng_particle_data_block_add(traj,
                                        TNG_TRAJ_VELOCITIES,
                                        "VELOCITIES",
                                        TNG_DOUBLE_DATA,
                                        TNG_TRAJECTORY_BLOCK,
                                        frames_in_chunk,
                                        3,
                                        1,
                                        0,
                                        NATOMS,
                                        TNG_TNG_COMPRESSION,
                                        vel_bytes)
            != TNG_SUCCESS)
        {
            free(pos_bytes);
            free(vel_bytes);
            error = "failed to add velocities block";
            goto cleanup;
        }
        if (tng_frame_set_write(traj, TNG_SKIP_HASH) != TNG_SUCCESS)
        {
            free(pos_bytes);
            free(vel_bytes);
            error = "failed to write frame set";
            goto cleanup;
        }

        free(pos_bytes);
        free(vel_bytes);
    }

cleanup:
    if (traj && tng_trajectory_destroy(&traj) != TNG_SUCCESS)
    {
        error = "failed to destroy trajectory";
    }
    if (error)
    {
        fail(error);
    }
    if (stat(path, &st) != 0)
    {
        fail("failed to stat output file");
    }
    return (uint64_t)st.st_size;
}

int main(int argc, char** argv)
{
    const size_t frame_len = NATOMS * 3u;
    const size_t total_values = NFRAMES * frame_len;
    const size_t reps = parse_reps(argc, argv);
    double*      positions = malloc(total_values * sizeof(*positions));
    double*      velocities = malloc(total_values * sizeof(*velocities));
    char*        output_path = 0;
    volatile uint64_t sink = 0;

    if (!positions || !velocities)
    {
        free(positions);
        free(velocities);
        fail("allocation failure");
    }

    output_path = bench_output_path();
    bench_data(positions, velocities);

    for (size_t rep = 0; rep < reps; rep++)
    {
        sink = bench_write(output_path, positions, velocities);
    }

    free(output_path);
    free(positions);
    free(velocities);

    return sink == 0;
}
