module purge
source ~/ofoamv9.sh
blockMesh
transformPoints "rotate=((1 0 0) (0 0 1))"
surfaceToPatch -tol 1e-2 circle.stl

export newmeshdir=$(foamListTimes -latestTime)
rm -rf constant/polyMesh/
cp -r $newmeshdir/polyMesh ./constant
rm -rf $newmeshdir
sed -i 's/patch0/solidsinlet/g' ./constant/polyMesh/boundary
sed -i 's/zone0/solidsinlet/g' ./constant/polyMesh/boundary

cp -r 0.orig 0
setFields
decomposePar -fileHandler collated
touch soln.foam
