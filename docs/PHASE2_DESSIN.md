# Phase 2 : le bras qui dessine

Branche : `phase-2`. À partir d'une image (PNG/JPG), le SO-ARM101 dessine les contours
avec un feutre serré dans la pince. Ça marche en sim (trace visible dans RViz) et ça
passera sur le vrai bras via `use_sim:=false`, une fois les deux paramètres de calibration réglés.

```
image.png ──► image_to_strokes ──► traits XY (m, repère robot)
                                        │
/draw_image (String) ──► brain_node ────┤ IK pinocchio (stylo vertical)
                                        ▼
                              /joint_command ──► driver (sim ou réel)
                              /drawing/plan, /drawing/ink ──► RViz
```

## Ce qui a été fait, point par point

### 0. Branche `phase-2`
Créée depuis `main`. `main` (pick & place) n'est pas modifiée.

### 1. Le feutre dans RViz : `ros2_ws/src/so101_description/urdf/so101.urdf`
- Pas de mesh à télécharger : deux primitives URDF (un cylindre orange Ø 12 mm × 120 mm
  et une bille noire au bout) dans un link `pen_link`, fixé à `gripper_frame_link`.
- Pourquoi `gripper_frame_link` : c'est le repère que l'IK utilise déjà. Son axe z pointe
  « hors de la pince », et l'IK `down=True` l'aligne sur -Z monde, donc le feutre est vertical.
- Un link `pen_tip_link` (à 3 cm sur z) marque le bout du feutre dans les TF.
- `robot_state_publisher` publie tout ça tout seul : aucun code ajouté.

### 2. Le feutre dans MuJoCo : `sim/so101_sim/assets/so101/so101_tek5.xml`
- Un `geom` cylindre dans le body `gripper`, avec la même géométrie que dans l'URDF.
- `contype=0 conaffinity=0` : visuel uniquement, il ne touche ni la table ni la pince, donc
  la physique ne bouge pas. On le voit dans l'image de la caméra externe.
- MuJoCo ne dessine pas de trait : la trace est affichée dans RViz (point 5).

### 3. Image → traits : `ros2_ws/src/so101_brain/so101_brain/image_to_strokes.py` (nouveau)
Module pur numpy + OpenCV, sans ROS, testable seul :
1. Image en niveaux de gris, redimensionnée à 400 px sur le plus grand côté.
2. Seuil d'Otsu : les formes sombres sur fond clair deviennent un masque binaire.
3. `cv2.findContours` récupère les contours, `cv2.approxPolyDP` les simplifie, et les contours de moins de 20 px (bruit) sont jetés.
4. Passage en mètres dans le repère robot : l'image est placée dans un carré de `draw_size`
   (10 cm) centré sur `draw_center` (0.22, 0.0), en gardant ses proportions. Le haut de
   l'image va vers +X (loin du robot), la droite de l'image vers -Y.
5. Ré-échantillonnage : un point tous les 4 mm, pour que l'IK suive bien les lignes droites.
6. Ordre des traits : on prend à chaque fois le trait le plus proche (glouton), et chaque
   boucle démarre au point le plus proche du stylo, pour limiter les trajets à vide.
- Auto-check : `python3 image_to_strokes.py` génère une image avec un cercle et un carré, puis vérifie
  qu'on obtient 2 traits, que tout est dans la zone, que les boucles sont fermées, que les
  points sont espacés d'au plus 4 mm et que l'orientation est bonne.

### 4. La machine à états du dessin : `ros2_ws/src/so101_brain/so101_brain/brain_node.py`
Elle réutilise ce qui existait : `_solve_ik(down=True)`, `_send_joint_command`, `_arrived`,
`_set_state` et `_move`. Le pick & place n'a pas changé.
- Nouveau topic `/draw_image` (`std_msgs/String`, chemin de l'image dans le container).
- Avant de commencer, on vérifie par IK les 4 coins de la feuille. Si l'un d'eux est hors de
  portée, on abandonne proprement.
- États : `PEN_TRAVEL` (au-dessus du début du trait, à `pen_lift` = 2 cm), puis `PEN_DOWN`,
  puis `TRACE` (on envoie le point suivant dès que le précédent est atteint à moins de
  `trace_tolerance` = 3 mm), puis `PEN_UP`, puis le trait suivant. À la fin : `RETURN`, puis `DONE`.
- Comme le stylo est vertical, la cible IK de `gripper_frame_link` est simplement le point
  du papier décalé de `pen_length` vers le haut (`_pen_target`). Aucune frame pinocchio n'a été ajoutée.
- La pince reste fermée tout le temps. `_go_home` prend maintenant un paramètre
  `gripper_pct`, pour qu'un retour maison (fin du dessin ou cible inatteignable) ne lâche pas le feutre.
- Paramètres ROS (ce sont les réglages de calibration) : `paper_z`, `pen_length`, `pen_lift`,
  `draw_center`, `draw_size`, `trace_tolerance`.

### 5. La trace dans RViz
- `/drawing/plan` (gris) : tous les traits prévus, publiés quand l'image est reçue.
- `/drawing/ink` (noir) : la position réelle du bout du feutre, calculée par FK à partir
  de `/joint_states`, enregistrée seulement quand il est à moins de 3 mm du papier. On voit donc ce qui est
  **vraiment** dessiné, pas ce qui était commandé.
- `ros2_ws/so101.rviz` : deux displays `Marker` ont été ajoutés.
- `so101_brain/package.xml` : la dépendance `visualization_msgs` a été ajoutée.

### 6. Lancement et Docker
- `brain.launch.py` accepte maintenant `autostart`, `paper_z` et `pen_length`.
- `docker/control_entrypoint.sh` et `docker-compose.yml` lisent la variable d'env `BRAIN_AUTOSTART`
  (par défaut `true`, comme avant). Avec `false`, le brain attend `/draw_image` au lieu de
  lancer le pick & place dès que l'objet est vu. Elle est documentée dans `.env.example`.
- Images de test : `ros2_ws/images/star.png` et `ros2_ws/images/smiley.png`.

## Utilisation (sim)

```bash
cd docker
docker-compose up -d control viz rviz        # RViz : http://localhost:6080/vnc.html
docker-compose exec control bash -lc 'source /ros2_ws/install/setup.bash && \
  ros2 topic pub -w 1 --once --keep-alive 5 /draw_image std_msgs/msg/String \
  "{data: /ros2_ws/images/smiley.png}"'
```

> `--keep-alive 5` est indispensable : sans lui, `ros2 topic pub --once` se ferme avant que
> le message soit livré (transport `LARGE_DATA`), et le brain ne reçoit rien.

## Résultats vérifiés (sim, Docker/colima)
- `image_to_strokes.py` auto-check : OK.
- Smiley : 5 traits, 241 points, dessinés en 27 s. Étoile : 7 traits, 224 points, en 27 s.
- Aucun timeout, aucun point sauté, aucune cible hors de portée : chaque point a été atteint
  à moins de 3 mm (`trace_tolerance`), avec une inclinaison du feutre de 0°.
- Dans RViz, le feutre est au bout de la pince et la trace noire dessine bien la forme.

## Passer au vrai bras
1. Serrer le feutre dans la pince, le bout environ 3 cm après les mors, et mesurer la longueur qui dépasse → `pen_length`.
2. Scotcher la feuille autour de (0.22, 0.0).
3. `paper_z` : envoyer `/target_position` à `z = paper_z + pen_length` et descendre par pas
   de 2 mm jusqu'à ce que le feutre touche juste le papier.
4. `USE_SIM=false BRAIN_AUTOSTART=false` dans `docker/.env`, puis lancer avec les
   valeurs mesurées (`ros2 param set /so101_brain paper_z ...`).
5. Si les servos n'arrivent jamais à 3 mm près (on verra des timeouts dans les logs), monter `trace_tolerance`.

## Limites connues / suite
- **Contours, pas squelette** : un trait épais dans l'image est dessiné deux fois (un passage
  par bord). Solution : squelettiser l'image (`cv2.ximgproc.thinning`) pour les dessins au trait.
- **Images en photo** : Otsu marche pour un dessin noir sur fond blanc. Pour une photo, il
  faudra Canny ou une étape de stylisation.
- **Vitesse** : on avance d'un point à la fois à 10 Hz, en attendant que chaque point soit atteint. C'est volontairement prudent ;
  pour aller plus vite, on peut monter `STATE_HZ` ou envoyer le point suivant un peu avant
  d'avoir atteint le précédent.
- **Papier plat supposé** : `paper_z` est le même partout. Si la table penche, il faudra mesurer
  3 coins et interpoler.
