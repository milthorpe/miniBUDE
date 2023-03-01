module BudeGPU {
  use Bude;

  // Energy evaluation parameters
  const CNSTNT: real(32) = 45.0;
  const HBTYPE_F: real(32) = 70.0;
  const HBTYPE_E: real(32) = 69.0;
  const HARDNESS: real(32) = 38.0;
  const NPNPDIST: real(32) = 5.5;
  const NPPDIST: real(32) = 1.0;

  /* Compute results with loaded data */
  proc compute(device: locale, results: [] real(32)) {
    writeln("\nRunning Chapel GPU");

    on device {
      var buffer: [0..<params.nposes] real(32);    
      // Copy data
      var poses = params.poses;
      var protein = params.protein;
      var ligand = params.ligand;
      var forcefield = params.forcefield;

      // Warm-up
      forall group in 0..<params.nposes/WGSIZE {
        fasten_main(params.natlig, params.natpro, protein, ligand,
                    poses, buffer, forcefield, group);
      }

      // Core part of computing
      const start = timestamp();
      for itr in 0..<params.iterations {
        forall group in 0..<params.nposes / WGSIZE {
          fasten_main(params.natlig, params.natpro, protein, ligand, poses, buffer, forcefield, group);
        }
      }
      const end = timestamp();

      // Copy to result
      results = buffer;
      printTimings(start, end);
    }
    
  }

  /* Core computing function */
  proc fasten_main(
    natlig: int,
    natpro: int,
    protein: [] atom,
    ligand: [] atom,
    transforms: [] real(32),
    results: [] real(32),
    forcefield: [] ffParams,
    group: int) {

    var transform: [0..<3, 0..<4, 0..<WGSIZE] real(32) = noinit;
    var etot: [0..<WGSIZE] real(32) = noinit;

    // Compute transformation matrix
    foreach i in 0..<WGSIZE {
      const ix = group*WGSIZE + i;
      const sx = sin(transforms(0, ix));
      const cx = cos(transforms(0, ix));
      const sy = sin(transforms(1, ix));
      const cy = cos(transforms(1, ix));
      const sz = sin(transforms(2, ix));
      const cz = cos(transforms(2, ix));
      transform(0, 0, i) = cy*cz;
      transform(0, 1, i) = sx*sy*cz - cx*sz;
      transform(0, 2, i) = cx*sy*cz + sx*sz;
      transform(0, 3, i) = transforms(3, ix);
      transform(1, 0, i) = cy*sz;
      transform(1, 1, i) = sx*sy*sz + cx*cz;      
      transform(1, 2, i) = cx*sy*sz - sx*cz;
      transform(1, 3, i) = transforms(4, ix);
      transform(2, 0, i) = -sy;
      transform(2, 1, i) = sx*cy;
      transform(2, 2, i) = cx*cy;
      transform(2, 3, i) = transforms(5, ix);

      etot[i] = 0.0;
    }
    
    foreach il in 0..<natlig {
      const l_atom = ligand[il];
      const l_params = forcefield[l_atom.aType];
      const lhphb_ltz = l_params.hphb < 0.0;
      const lhphb_gtz = l_params.hphb > 0.0;

      // Transform ligand atom
      var lpos_x: [0..WGSIZE] real(32) = noinit;
      var lpos_y: [0..WGSIZE] real(32) = noinit;
      var lpos_z: [0..WGSIZE] real(32) = noinit;

      foreach l in 0..<WGSIZE {
        lpos_x[l] = transform(0, 3, l)
          + l_atom.x * transform(0, 0, l)
          + l_atom.y * transform(0, 1, l)
          + l_atom.z * transform(0, 2, l);

        lpos_y[l] = transform(1, 3, l)
          + l_atom.x * transform(1, 0, l)
          + l_atom.y * transform(1, 1, l)
          + l_atom.z * transform(1, 2, l);

        lpos_z[l] = transform(2, 3, l)
          + l_atom.x * transform(2, 0, l)
          + l_atom.y * transform(2, 1, l)
          + l_atom.z * transform(2, 2, l);
      }

      foreach ip in 0..<natpro {
        const p_atom = protein(ip);
        const p_params = forcefield(p_atom.aType);

        const radij = p_params.radius + l_params.radius;
        const r_radij = 1.0 / radij;

        const elcdst = if
          p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F
          then 4.0: real(32)
          else 2.0: real(32);

        const elcdst1 = if
          p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F
          then 0.25: real(32)
          else 0.5: real(32);

        const type_E = p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E;
        const phphb_ltz = p_params.hphb <  0;
        const phphb_gtz = p_params.hphb >  0;
        const phphb_nz  = p_params.hphb != 0;

        const p_hphb = p_params.hphb 
          * if phphb_ltz && lhphb_gtz then -1.0: real(32) else 1.0: real(32);

        const l_hphb = l_params.hphb 
          * if phphb_gtz && lhphb_ltz then -1.0: real(32) else 1.0: real(32);

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

        const r_distdslv = 1.0 / distdslv;
        const chrg_init = l_params.elsc * p_params.elsc;
        const dslv_init = p_hphb + l_hphb; 

        foreach l in 0..<WGSIZE {
          // Calculate distance between atoms
          const x = lpos_x(l) - p_atom.x;
          const y = lpos_y(l) - p_atom.y;
          const z = lpos_z(l) - p_atom.z;
          const distij = sqrt(x * x + y * y + z* z); 

          // Calculate the sum of the sphere radii
          const distbb = distij - radij;
          const zone1 = distbb < 0.0: real(32);

          // Calculate steric energy
          etot[l] += (1.0 - distij * r_radij)
            * if zone1 then 2.0: real(32) * HARDNESS else 0.0: real(32);

          // Calculate formal and dipole charge interactions
          var chrg_e =
            chrg_init * (
              if zone1 
              then 1.0: real(32)
                else 1.0: real(32) - distbb * elcdst1
            ) * (
              if distbb < elcdst 
              then 1.0: real(32)
              else 0.0: real(32)
            );
          
          var neg_chrg_e = -abs(chrg_e);
          chrg_e = if type_E then neg_chrg_e else chrg_e;
          etot[l] += chrg_e * CNSTNT;

          const coeff = 1.0 - distbb * r_distdslv;
          var dslv_e = dslv_init 
            * if distbb < distdslv && phphb_nz then 1.0: real(32) else 0.0: real(32);

          dslv_e *= if zone1 then 1.0: real(32) else coeff;
          etot[l] += dslv_e;
        }
      }
    }

    results[group * WGSIZE..<(group + 1) * WGSIZE] = 0.5 : real(32) * etot;
  }
}