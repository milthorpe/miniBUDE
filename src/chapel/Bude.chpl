module Bude {
  use Time;
  use IO;
  use GPU;
  use Math;
  use CTypes;
  use ArgumentParser;
  use Path;
  use ChplConfig;

  config param VERSION_STRING = "2.0";
  config param MINIBUDE_COMPILE_COMMANDS = "";
  config param useGPU = false;

  // Program context parameters
  param DEFAULT_ITERS = 8;
  param DEFAULT_NPOSES = 65536;
  param DEFAULT_WGSIZE = 1;
  param REF_NPOSES = 65536;
  param DATA_DIR = "../../data/bm1";
  param FILE_LIGAND = "/ligand.in";
  param FILE_PROTEIN = "/protein.in";
  param FILE_FORCEFIELD = "/forcefield.in";
  param FILE_POSES = "/poses.in";
  param FILE_REF_ENERGIES = "/ref_energies.out";

  // Energy evaluation parameters
  param CNSTNT: real(32) = 45.0;
  param HBTYPE_F: real(32) = 70.0;
  param HBTYPE_E: real(32) = 69.0;
  param HARDNESS: real(32) = 38.0;
  param NPNPDIST: real(32) = 5.5;
  param NPPDIST: real(32) = 1.0;
  param ZERO: real(32) = 0.0;
  param QUARTER: real(32) = 0.25;
  param HALF: real(32) = 0.5;
  param ONE: real(32) = 1.0;
  param TWO: real(32) = 2.0;
  param FOUR: real(32) = 4.0;

  // Configurations
  config param PPWI: int(32) = 4; // "poses per work item" = work per Chapel task
  // TODO allow multiple choices for PPWI

  record atom {
    var x, y, z: real(32);
    var aType: int(32);
  }

  record ffParams {
    var hbtype: int(32);
    var radius: real(32);
    var hphb: real(32);
    var elsc: real(32);
  }

  class params {
    var deckDir: string;
    var iterations: int;
    var natlig, natpro, ntypes, nposes: int;
    var wgsize: int;

    var proteinDomain: domain(1);
    var ligandDomain: domain(1);
    var forcefieldDomain: domain(1);
    var posesDomain: domain(2);

    var protein: [proteinDomain] atom; 
    var ligand: [ligandDomain] atom;
    var forcefield: [forcefieldDomain] ffParams;
    var poses: [posesDomain] real(32);

    var list: bool;
    var deviceIndex: int(32);

    proc init() { }

    proc load(args: [] string) throws {
      // Parsing command-line parameters
      var parser = new argumentParser();

      var listArg = parser.addFlag(name="list",
        opts=["-l", "--list"],
        defaultValue=false,
        help="List available devices");

      var deviceArg = parser.addOption(name="device",
        opts=["-d", "--device"],
        defaultValue=0: string,
        valueName="INDEX",
        help="""Select device at INDEX from output of --list
                        [optional] default=0""");
      var iterationsArg = parser.addOption(name="iterations",
        opts=["-i", "--iter"],
        defaultValue=DEFAULT_ITERS: string,
        valueName="I",
        help="""Repeat kernel I times\n
                        [optional] default=""" + DEFAULT_ITERS: string);
      var numposesArg = parser.addOption(name="numposes",
        opts=["-n", "--poses"],
        defaultValue=DEFAULT_NPOSES: string,
        valueName="N",
        help="""Compute energies for only N poses, use 0 for deck max
                        [optional] default=0""");
      var wgsizeArg = parser.addOption(name="wgsize",
        opts=["-w", "--wgsize"],
        defaultValue=DEFAULT_WGSIZE: string,
        valueName="WGSIZE",
        help="""A CSV list of work-group sizes, not all implementations support this parameter
                        [optional] default=""" + DEFAULT_WGSIZE: string);
      var deckArg = parser.addOption(name="deck",
        opts=["--deck"],
        defaultValue=DATA_DIR,
        valueName="DIR",
        help="""Use the DIR directory as input deck
                        [optional] default=""" + DATA_DIR);
      parser.parseArgs(args);

      // Store these parameters
      this.list = listArg.valueAsBool();

      try {
        this.iterations = iterationsArg.value(): int(32);
        if (this.iterations < 0) then throw new Error();
      } catch {
        writeln("Invalid number of iterations");
        exit(1);
      }

      try {
        this.deviceIndex = deviceArg.value(): int(32);
        if (this.deviceIndex < 0) then throw new Error();
      } catch {
        writeln("Invalid device index");
        exit(1);
      }

      try {
        this.nposes = numposesArg.value(): int(32);
        if (this.nposes < 0) then throw new Error();
      } catch {
        writeln("Invalid number of poses");
        exit(1);
      }

      try {
        this.wgsize = wgsizeArg.value(): int;
        if (this.wgsize < 0) then throw new Error();
      } catch {
        writeln("Invalid number of blocks");
        exit(1);
      }
      
      this.deckDir = deckArg.value(); 
      
      // Load data
      var length: int;
      var aFile: file;
      var reader: fileReader(locking=false, ?);
      
      // Read ligand
      aFile = openFile(this.deckDir + FILE_LIGAND, length);
      this.natlig = length / c_sizeof(atom): int;
      this.ligandDomain = {0..<this.natlig};
      reader = aFile.reader(region=0.., deserializer=new binaryDeserializer());
      reader.read(this.ligand);
      reader.close(); aFile.close();

      // Read protein
      aFile = openFile(this.deckDir + FILE_PROTEIN, length);
      this.natpro = length / c_sizeof(atom): int;
      this.proteinDomain = {0..<this.natpro};
      reader = aFile.reader(region=0.., deserializer=new binaryDeserializer());
      reader.read(this.protein);
      reader.close(); aFile.close();

      // Read force field
      aFile = openFile(this.deckDir + FILE_FORCEFIELD, length);
      this.ntypes = length / c_sizeof(ffParams): int;
      this.forcefieldDomain = {0..<this.ntypes};
      reader = aFile.reader(region=0.., deserializer=new binaryDeserializer());
      reader.read(this.forcefield);
      reader.close(); aFile.close();

      // Read poses
      this.posesDomain = {0..<6, 0..<this.nposes};
      aFile = openFile(this.deckDir + FILE_POSES, length);
      reader = aFile.reader(region=0.., deserializer=new binaryDeserializer());
      var available = (length / (6 * c_sizeof(real(32)): int));
      var current = 0;
      while (current < this.nposes) {
        var fetch = this.nposes - current;
        if (fetch > available) then fetch = available;

        for i in posesDomain.dim(0) {
          const base = i*available*c_sizeof(real(32)):int;
          const amount = fetch*c_sizeof(real(32)):int;
          reader.seek(region=base..base+amount);
          reader.read(this.poses(i, current..));
        }
        current += fetch;
      }
      this.nposes = current;
      reader.close(); aFile.close();
    }
  }

  var context: params = new params();

  extern proc get_device_driver_version(const deviceIndex: int(32)): int(32);
  extern proc get_device_name(const deviceIndex: int(32)): c_ptrConst(c_char);

  proc getDeviceName(deviceIndex: int(32)): string {
      const deviceName = get_device_name(deviceIndex);
      return try! string.createBorrowingBuffer(deviceName);
  }

  proc enumerateDevices() {
    if useGPU {
      return [deviceId in 0..#here.gpus.size] (deviceId, getDeviceName(deviceId: int(32)));
    } else {
      return [(0, "Chapel CPU")];
    }
  }

  proc main(args: [] string) {
    try! context.load(args);

    writeln("miniBUDE:  ", VERSION_STRING);
    writeln("compile_commands:");
    writeln("   - \"", MINIBUDE_COMPILE_COMMANDS, "\"");
    // TODO VCS info
    // TODO CPU info

    writeln("~");
    const now = timeSinceEpoch().totalSeconds();
    writef("time: { epoch_s: %i, formatted: \"%s\" }\n", now, dateTime.createFromTimestamp(now):string);

    writeln("deck:");
    writeln("  path:         \"", context.deckDir, "\"");
    writeln("  poses:        ", context.nposes); // TODO maxposes
    writeln("  proteins:     ", context.natpro);
    writeln("  ligands:      ", context.natlig);
    writeln("  forcefields:  ", context.ntypes);
    writeln("config:");
    writeln("  iterations:   ", context.iterations);
    writeln("  poses:        ", context.nposes);
    writeln("  ppwi:");
    writeln("    available:  [", PPWI, "]");
    writeln("    selected:   [", PPWI, "]");
    writeln("  wgsize:       [", context.wgsize, "]");

    const devices = enumerateDevices();
    if devices.size == 0 {
      try! stderr.writeln(" # (no devices available)");
    } else {
      if context.list {
        writeln("devices:");
        for device in devices {
          writeln("  ", device(0), ": \"", device(1), "\"");
        }
        exit(0);
      } else {
        if context.deviceIndex >= 0 && context.deviceIndex < devices.size {
          writeln("device: { index: ", context.deviceIndex, ", name: \"", devices(context.deviceIndex)(1), "\" }");
        } 
      }
    }

    // Compute
    var energies: [0..<context.nposes] real(32);
    compute(energies);

    // Validate
    var length: int;
    const ref_energies = openFile(context.deckDir+FILE_REF_ENERGIES, length);
    var e: real(32);
    var diff: real(32);
    var maxdiff: real(32) = -100.0;
    var n_ref_poses = context.nposes;
    if (context.nposes > REF_NPOSES) {
      writeln("Only validating the first ", REF_NPOSES, " poses");
      n_ref_poses = REF_NPOSES;
    }
    var reader = try! ref_energies.reader();
    for i in 0..<n_ref_poses {
      try! reader.read(e);
      if (abs(e) < 1.0 && abs(energies(i)) < 1.0) {
        continue;
      }
      diff = abs(e - energies(i)) / e;
      if (diff > maxdiff) {
        maxdiff = diff;
      }
    }
    writef("\nLargest difference was %{.###}%%.\n\n", 100 * maxdiff);
  } // main

  proc compute(ref results: [] real(32)) {
    if (CHPL_GPU == "nvidia" || CHPL_GPU == "amd") {
      writeln("\nRunning Chapel on ", here.gpus.size, (if here.gpus.size > 1 then " GPUs" else " GPU"));
      gpukernel(context, results);
    } else if (CHPL_GPU == "none") {
      writeln("\nRunning Chapel");
      cpukernel(context, results);
    } else {
      writeln("\n" + CHPL_GPU + " is not supported.");
      exit(1);
    }
  } 

  proc gpukernel(context: params, results: [] real(32)) {
    const ngpu = here.gpus.size;
    var times: [0..<ngpu] real;
    coforall (gpu, gpuID) in zip(here.gpus, here.gpus.domain) with (ref times) do on gpu {
      const iterations = context.iterations: int(32);
      const nposes = (context.nposes / ngpu) : int(32);
      const natlig = context.natlig: int(32);
      const natpro = context.natpro: int(32);
      const wgsize = context.wgsize: int(32);

      const protein = context.protein;
      const ligand = context.ligand;
      const forcefield = context.forcefield;
      const poses: [0:int(32)..<6:int(32), 0..<nposes] real(32) = context.poses[{0..<6, gpuID*nposes..<(gpuID+1)*nposes}];
      var buffer: [0..<nposes] real(32);

      times[gpuID] = timestampMS();
      for i in 0..<iterations {
        @assertOnGpu foreach group in 0..<nposes/PPWI {
          __primitive("gpu set blockSize", wgsize);
          fasten_main(natlig, natpro, protein, ligand, 
                      poses, buffer, forcefield, group: int(32));

        }
      }
      results[gpuID*nposes..<(gpuID+1)*nposes] = buffer;
      times[gpuID] = timestampMS() - times[gpuID];
    }

    printTimings(max reduce times);
  }

  proc cpukernel(ref context: params, ref results: [] real(32)) {
    var buffer: [0..<context.nposes] real(32); 
    var poses = context.poses;
    var protein = context.protein;
    var ligand = context.ligand;
    var forcefield = context.forcefield;

    const natlig = context.natlig: int(32);
    const natpro = context.natpro: int(32);
    const nposes = context.nposes: int(32);

    // Warm-up
    forall group in 0..<nposes/PPWI {
      fasten_main(natlig, natpro, protein, ligand,
                  poses, buffer, forcefield, group: int(32));
    }

    // Core part of computing
    const start: real = timestampMS();
    for itr in 0..<context.iterations {
      forall group in 0..<nposes/PPWI {
        fasten_main(natlig, natpro, protein, ligand,
                  poses, buffer, forcefield, group: int(32));
      }
    }
    const end: real = timestampMS();

    // Copy to result
    results = buffer;

    printTimings(end - start);
  }

  private inline proc fasten_main(
    const in natlig: int(32),
    const in natpro: int(32),
    const ref protein: [] atom,
    const ref ligand: [] atom,
    const ref transforms: [] real(32),
    ref results: [] real(32),
    const ref forcefield: [] ffParams,
    const in group: int(32)) {

    const offset = group * PPWI;
    var etot: PPWI * real(32);
    var transform: 3 * (4 * (PPWI * real(32)));

    // Compute transformation matrix
    for param i in 0:int(32)..<PPWI {
      const ix = offset + i;
      const sx = sin(transforms(0, ix));
      const cx = cos(transforms(0, ix));
      const sy = sin(transforms(1, ix));
      const cy = cos(transforms(1, ix));
      const sz = sin(transforms(2, ix));
      const cz = cos(transforms(2, ix));
      transform[0][0][i] = cy*cz;
      transform[0][1][i] = sx*sy*cz - cx*sz;
      transform[0][2][i] = cx*sy*cz + sx*sz;
      transform[0][3][i] = transforms(3, ix);
      transform[1][0][i] = cy*sz;
      transform[1][1][i] = sx*sy*sz + cx*cz;      
      transform[1][2][i] = cx*sy*sz - sx*cz;
      transform[1][3][i] = transforms(4, ix);
      transform[2][0][i] = -sy;
      transform[2][1][i] = sx*cy;
      transform[2][2][i] = cx*cy;
      transform[2][3][i] = transforms(5, ix);

      etot[i] = ZERO;
    }
    
    // Loop over ligand atoms
    for il in 0..<natlig {
      // Load ligand atom data
      const l_atom = ligand[il];
      const l_params = forcefield[l_atom.aType];
      const lhphb_ltz = l_params.hphb < ZERO;
      const lhphb_gtz = l_params.hphb > ZERO;

      // Transform ligand atom
      var lpos_x: PPWI * real(32);
      var lpos_y: PPWI * real(32);
      var lpos_z: PPWI * real(32);

      for param i in 0:int(32)..<PPWI {
        lpos_x[i] = transform[0][3][i]
          + l_atom.x * transform[0][0][i]
          + l_atom.y * transform[0][1][i]
          + l_atom.z * transform[0][2][i];

        lpos_y[i] = transform[1][3][i]
          + l_atom.x * transform[1][0][i]
          + l_atom.y * transform[1][1][i]
          + l_atom.z * transform[1][2][i];

        lpos_z[i] = transform[2][3][i]
          + l_atom.x * transform[2][0][i]
          + l_atom.y * transform[2][1][i]
          + l_atom.z * transform[2][2][i];
      }

    // Loop over protein atoms
      for ip in 0..<natpro {
        // Load protein atom data
        const p_atom = protein(ip);
        const p_params = forcefield(p_atom.aType);

        const radij = p_params.radius + l_params.radius;
        const r_radij = ONE / radij;

        const elcdst = if
          p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F
          then FOUR
          else TWO;

        const elcdst1 = if
          p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F
          then QUARTER
          else HALF;

        const type_E = p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E;
        const phphb_ltz = p_params.hphb <  0;
        const phphb_gtz = p_params.hphb >  0;
        const phphb_nz  = p_params.hphb != 0;

        const p_hphb = p_params.hphb 
          * if phphb_ltz && lhphb_gtz then -ONE else ONE;

        const l_hphb = l_params.hphb 
          * if phphb_gtz && lhphb_ltz then -ONE else ONE;

        const distdslv =
          if phphb_ltz
          then (
            if lhphb_ltz
            then NPNPDIST
            else NPPDIST
          ) else (
            if lhphb_ltz
            then NPPDIST
            else -max(real(32))
          );

        const r_distdslv = ONE / distdslv;
        const chrg_init = l_params.elsc * p_params.elsc;
        const dslv_init = p_hphb + l_hphb; 

        for param i in 0:int(32)..<PPWI {
          // Calculate distance between atoms
          const x = lpos_x[i] - p_atom.x;
          const y = lpos_y[i] - p_atom.y;
          const z = lpos_z[i] - p_atom.z;
          const distij = sqrt(x * x + y * y + z* z); 

          // Calculate the sum of the sphere radii
          const distbb = distij - radij;
          const zone1 = distbb < ZERO;

          // Calculate steric energy
          etot[i] += (ONE - distij * r_radij)
            * if zone1 then TWO * HARDNESS else ZERO;

          // Calculate formal and dipole charge interactions
          var chrg_e =
            chrg_init * (
              if zone1 
              then ONE
                else ONE - distbb * elcdst1
            ) * (
              if distbb < elcdst 
              then ONE
              else ZERO
            );
          
          const neg_chrg_e = -abs(chrg_e);
          chrg_e = if type_E then neg_chrg_e else chrg_e;
          etot[i] += chrg_e * CNSTNT;

          const coeff = ONE - distbb * r_distdslv;
          var dslv_e = dslv_init 
            * if distbb < distdslv && phphb_nz then ONE else ZERO;

          dslv_e *= if zone1 then ONE else coeff;
          etot[i] += dslv_e;
        }
      }
    }

    for param i in 0:int(32)..<PPWI {
      results[offset+i] = etot[i] * HALF;
    }
  }

  proc openFile(fileName: string, ref length: int): file {
    try {
      const aFile = open(fileName, ioMode.r);
      length = aFile.size;
      return aFile;
    } catch {
      try! stderr.writeln("Failed to open '", fileName, "'");
      exit(0);
    }
  }

  proc timestampMS() {
    return timeSinceEpoch().totalSeconds() * 1000;
  }

  proc printTimings(timeMS: real(64)) {
    const ms = timeMS / context.iterations;
    const runtime = ms * 1e-3;

    const ops_per_wg = PPWI * 27 + context.natlig * (2 + PPWI * 18 + context.natpro * (10 + PPWI * 30)) + PPWI;
    const total_ops = ops_per_wg * (context.nposes / PPWI);
    const flops = total_ops / runtime;
    const gflops = flops / 1e9;

    const interactions = 1.0 * context.nposes * context.natlig * context.natpro;
    const interactions_per_sec = interactions / runtime;

    // Print stats
    writef("- Total time:     %7.3dr ms\n", timeMS);
    writef("- Average time:   %7.3dr ms\n", ms);
    writef("- Interactions/s: %7.3dr billion\n", (interactions_per_sec / 1e9));
    writef("- GFLOP/s:        %7.3dr\n", gflops);
  }
}
