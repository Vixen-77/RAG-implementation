import {User ,registerValidation , loginValidation} from "../models/UserSchema.js"
import expressAsyncHandler from "express-async-handler"
import bcrypt from "bcryptjs"
import jwt from "jsonwebtoken";


export const UserRegister = expressAsyncHandler(async(req,res)=>{
    const {error} = registerValidation(req.body)
    if(error){
        return res.status(400).json({message:error.details[0].message})
    }

    const user = await User.findOne({email:req.body.email})
    if(user){
        return res.status(400).json({message:"this user already exist , Login"})
    }
    const salt = await bcrypt.genSalt(12)
    const newPassword = await bcrypt.hash(req.body.password ,salt);

    const newUser = await User.create({
        username:req.body.username,
        email:req.body.email,
        password:newPassword,
    })
    const token = jwt.sign({ id: newUser._id }, process.env.JWT_SECRET, { expiresIn: "30d" });

    return res.status(201).json({
        message: "user created",
        token,
        user: {
            id: newUser._id,
            username: newUser.username,
            email: newUser.email
        }
    })
})

export const UserLogin = expressAsyncHandler(async(req,res)=>{
    const {error} = loginValidation(req.body)
    if(error){
        return res.status(400).json({message:error.details[0].message})
    }
    const user = await User.findOne({email:req.body.email})
    if(!user){
        return res.status(400).json({message:"incorrect email or password"})
    }
    const isPasswordMatch = await bcrypt.compare(req.body.password , user.password)
     if(!isPasswordMatch){
        return res.status(400).json({message:"incorrect email or password"})
    }
    return res.status(200).json({
        message: "login success",
        token: jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: "30d" }),
        user: {
            id: user._id,
            username: user.username,
            email: user.email
        }
    })
})