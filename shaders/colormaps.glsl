// Polynomial fits of perceptual colormaps (matplotlib inferno/viridis and
// Google's turbo). Input t in [0,1], output sRGB-encoded colour.
// The same formulas live in src/colormap.h for the GIF palette.

vec3 cmInferno(float t) {
    const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
    const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
    const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
    const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
    const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
    const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
    const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

vec3 cmViridis(float t) {
    const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

vec3 cmTurbo(float t) {
    const vec4 kRed4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreen4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlue4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRed2 = vec2(-152.94239396, 59.28637943);
    const vec2 kGreen2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlue2 = vec2(-89.90310912, 27.34824973);
    vec4 v4 = vec4(1.0, t, t * t, t * t * t);
    vec2 v2 = v4.zw * v4.z;
    return vec3(dot(v4, kRed4) + dot(v2, kRed2),
                dot(v4, kGreen4) + dot(v2, kGreen2),
                dot(v4, kBlue4) + dot(v2, kBlue2));
}

vec3 colormap(int which, float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c;
    if (which == 1) c = cmViridis(t);
    else if (which == 2) c = cmTurbo(t);
    else if (which == 3) c = vec3(t);
    else c = cmInferno(t);
    return clamp(c, 0.0, 1.0);
}
